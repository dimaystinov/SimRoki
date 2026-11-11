// Continuous context-cell readout trained by official KNP resource STDP.
// stdin: target initial_weight pairs, one per context cell. No external weight updates.
#include <knp/backends/cpu-single-threaded/backend.h>
#include <knp/framework/backend_loader.h>
#include <knp/core/messaging/spike_message.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <variant>
#include <cstdio>
#include <unistd.h>

using Plastic = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using Delta = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using Population = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;

int main(int argc,char** argv) {
    const unsigned epochs=argc>1?std::stoul(argv[1]):120;
    std::vector<double> targets,initial;
    double target,weight;
    while(std::cin>>target>>weight) {
        // Targets use the task range; checkpoints use the synapse's physical
        // (0,1) range. KNP can export 0.049999997 after fitting target 0.05.
        if(!std::isfinite(target)||!std::isfinite(weight)||target<0.05||target>0.8||weight<=0.0||weight>=1.0) {
            std::cerr<<"target must be finite in [0.05,0.8], initial weight in (0,1)\n";
            return 1;
        }
        targets.push_back(target);initial.push_back(weight);
    }
    if(targets.empty()) throw std::runtime_error("no training cells");
    // The packaged debug backend logs every synapse. Keep the JSON protocol compact.
    const int result_fd=dup(STDOUT_FILENO);
    if(result_fd<0 || !freopen("/dev/null","w",stdout)) throw std::runtime_error("cannot isolate backend log");
    Population population{knp::core::UID{},[](size_t){
        Population::NeuronParameters p;
        p.activation_threshold_=0.01;
        p.d_h_=0; // isolate dopamine plasticity from unsupervised drift
        p.dopamine_plasticity_time_=6;
        p.isi_max_=2;
        p.free_synaptic_resource_=0;
        return p;
    },targets.size()};
    Plastic plastic{knp::core::UID{false},population.get_uid(),[&](size_t i)->std::optional<Plastic::Synapse>{
        Plastic::SynapseParameters p;
        p.weight_=initial[i];p.delay_=1;p.output_type_=knp::synapse_traits::OutputType::EXCITATORY;
        p.rule_.synaptic_resource_=initial[i]/(1-initial[i]);
        p.rule_.w_min_=0;p.rule_.w_max_=1;p.rule_.dopamine_plasticity_period_=6;
        return Plastic::Synapse{p,i,i};
    },targets.size()};
    plastic.unlock_weights();
    Delta dopamine{knp::core::UID{false},population.get_uid(),[](size_t i)->std::optional<Delta::Synapse>{
        return Delta::Synapse{{0,1,knp::synapse_traits::OutputType::DOPAMINE},i,i};
    },targets.size()};
    knp::framework::BackendLoader loader;
    auto backend=loader.load("/usr/lib/libknp-cpu-single-threaded-backend.so");
    auto* cpu=dynamic_cast<knp::backends::single_threaded_cpu::SingleThreadedCPUBackend*>(backend.get());
    if(!cpu) throw std::runtime_error("wrong backend");
    backend->load_all_populations({population});backend->load_all_projections({plastic,dopamine});backend->_init();
    backend->start_learning();
    Plastic* learned=nullptr;Delta* reward=nullptr;
    for(auto it=cpu->begin_projections();it!=cpu->end_projections();++it){
        if(auto* p=std::get_if<Plastic>(&it->arg_)) learned=p;
        if(auto* p=std::get_if<Delta>(&it->arg_)) reward=p;
    }
    if(!learned||!reward)throw std::runtime_error("missing projections");
    knp::core::UID sensory_uid,reward_uid;
    backend->subscribe<knp::core::messaging::SpikeMessage>(plastic.get_uid(),{sensory_uid});
    backend->subscribe<knp::core::messaging::SpikeMessage>(dopamine.get_uid(),{reward_uid});
    auto endpoint=backend->get_message_bus().create_endpoint();
    std::vector<knp::core::messaging::SpikeIndex> indexes(targets.size());
    std::iota(indexes.begin(),indexes.end(),0);
    for(unsigned epoch=0;epoch<epochs;++epoch){
        for(size_t i=0;i<targets.size();++i){
            // Only the non-plastic dopamine signal is set by the teacher.
            const auto current=std::get<knp::core::synapse_data>((*learned)[i]).weight_;
            std::get<knp::core::synapse_data>((*reward)[i]).weight_=0.5*(targets[i]-current);
        }
        for(unsigned local=0;local<12;++local){
            const auto step=backend->get_step();
            if(local==0)endpoint.send_message(knp::core::messaging::SpikeMessage{{sensory_uid,step},indexes});
            if(local==3)endpoint.send_message(knp::core::messaging::SpikeMessage{{reward_uid,step},indexes});
            backend->_step();
        }
    }
    double max_error=0,change=0;
    fflush(stdout);dup2(result_fd,STDOUT_FILENO);close(result_fd);
    std::cout<<std::setprecision(17)<<"{\"backend\":\"KNP resource STDP\",\"epochs\":"<<epochs<<",\"weights\":[";
    for(size_t i=0;i<targets.size();++i){
        const auto w=std::get<knp::core::synapse_data>((*learned)[i]).weight_;
        if(i)std::cout<<',';std::cout<<w;
        max_error=std::max(max_error,std::abs(w-targets[i]));change+=std::abs(w-initial[i]);
    }
    std::cout<<"],\"max_error\":"<<max_error<<",\"weight_change\":"<<change<<"}\n";
    return max_error<1e-5?0:2;
}
