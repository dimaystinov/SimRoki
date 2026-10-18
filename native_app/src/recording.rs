use macroquad::prelude::*;
use std::{io::Write, path::PathBuf, process::{Command, Stdio}, sync::mpsc, time::{Instant, SystemTime, UNIX_EPOCH}};

const FPS: f64 = 30.0;
#[derive(Clone, Copy)]
pub struct AudioState { pub playing: bool, pub position: f64, pub volume: f32 }
#[derive(Clone)]
struct Segment { start: f64, position: f64, duration: f64, volume: f32 }
enum Message { Frame(u64, Image), Finish(u64, Vec<Segment>) }
struct Active {
    start: Instant, next_frame: u64, tx: mpsc::SyncSender<Message>,
    segments: Vec<Segment>, audio: Option<(f64, AudioState)>,
}
#[derive(Default)]
pub struct Recorder {
    active: Option<Active>, result: Option<mpsc::Receiver<Result<PathBuf, String>>>,
    status: String,
}
impl Recorder {
    pub fn frame(&mut self, audio: AudioState, neon: bool) {
        if let Some(rx) = &self.result {
            if let Ok(result) = rx.try_recv() {
                self.status = match result { Ok(p) => format!("Saved: {}", p.display()), Err(e) => format!("Recording error: {e}") };
                self.result = None;
                self.active = None;
            }
        }
        let button = Rect::new(screen_width()-52.0, screen_height()-66.0, 32.0, 32.0);
        let hovered = button.contains(vec2(mouse_position().0, mouse_position().1));
        if hovered && is_mouse_button_pressed(MouseButton::Left) {
            if let Some(mut active) = self.active.take() {
                let elapsed = active.start.elapsed().as_secs_f64();
                close_segment(&mut active, elapsed);
                let _ = active.tx.send(Message::Finish((elapsed*FPS).ceil() as u64, active.segments));
                self.status = "Saving MP4...".into();
            } else if self.result.is_none() {
                match start_worker() {
                    Ok((tx, rx)) => {
                        self.active = Some(Active { start: Instant::now(), next_frame: 0, tx, segments: vec![], audio: if audio.playing {Some((0.0,audio))} else {None} });
                        self.result = Some(rx); self.status.clear();
                    },
                    Err(e) => self.status = e,
                }
            }
        }
        if let Some(active) = &mut self.active {
            let elapsed = active.start.elapsed().as_secs_f64();
            let was_playing = active.audio.is_some();
            let changed_volume = active.audio.is_some_and(|(_, a)| a.volume != audio.volume);
            if was_playing && (!audio.playing || changed_volume) { close_segment(active, elapsed); }
            if audio.playing && active.audio.is_none() { active.audio = Some((elapsed,audio)); }
            let frame = (elapsed*FPS).floor() as u64;
            if frame >= active.next_frame {
                let image = get_screen_data();
                // Screen readback binds a raw OpenGL texture outside miniquad's
                // cache. In the GL backend commit_frame clears those bindings,
                // so subsequent UI draws bind their actual textures again.
                unsafe { macroquad::window::get_internal_gl().quad_context.commit_frame(); }
                match active.tx.try_send(Message::Frame(frame, image)) {
                    Ok(()) => active.next_frame = frame+1,
                    Err(mpsc::TrySendError::Full(_)) => {},
                    Err(mpsc::TrySendError::Disconnected(_)) => { self.status = "Encoder stopped".into(); }
                }
            }
        }

        let ink = if neon {WHITE} else {DARKGRAY};
        draw_rectangle_lines(button.x,button.y,button.w,button.h,1.0,ink);
        if self.active.is_some() { draw_rectangle(button.x+9.0,button.y+9.0,14.0,14.0,RED); }
        else { draw_circle(button.x+16.0,button.y+16.0,8.0,RED); }
        let label = if let Some(active) = &self.active { format!("REC {:.0}s",active.start.elapsed().as_secs_f64()) }
            else if self.result.is_some() { "Saving MP4...".into() }
            else if hovered { "Record MP4 + app audio".into() }
            else { self.status.clone() };
        if !label.is_empty() {
            let width=measure_text(&label,None,16,1.0).width;
            draw_text(&label,(screen_width()-width-16.0).max(12.0),screen_height()-12.0,16.0,ink);
        }

    }
}
fn close_segment(active: &mut Active, now: f64) {
    if let Some((start,a))=active.audio.take() {
        if now>start { active.segments.push(Segment{start,position:a.position,duration:now-start,volume:a.volume}); }
    }
}
fn ffmpeg() -> Result<PathBuf,String> {
    for path in [PathBuf::from("ffmpeg"),PathBuf::from("/opt/homebrew/bin/ffmpeg"),PathBuf::from("/usr/local/bin/ffmpeg")] {
        if Command::new(&path).arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s|s.success()) { return Ok(path); }
    }
    Err("MP4 recording requires FFmpeg".into())
}
fn start_worker() -> Result<(mpsc::SyncSender<Message>,mpsc::Receiver<Result<PathBuf,String>>),String> {
    let encoder=ffmpeg()?;
    let home=std::env::var_os("HOME").or_else(||std::env::var_os("USERPROFILE")).ok_or("Home directory unavailable")?;
    let dir=PathBuf::from(home).join(if cfg!(target_os="windows") {"Videos"} else {"Movies"}).join("SimRoki");
    std::fs::create_dir_all(&dir).map_err(|e|e.to_string())?;
    let id=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let output=dir.join(format!("SimRoki-{id}.mp4"));
    let temp=dir.join(format!(".recording-{id}"));
    std::fs::create_dir(&temp).map_err(|e|e.to_string())?;
    let (tx,rx)=mpsc::sync_channel(3); let (done,result)=mpsc::channel();
    std::thread::spawn(move || {
        let result=encode(&encoder,&temp,&output,rx).map(|_|output);
        if result.is_ok() { let _=std::fs::remove_dir_all(&temp); }
        let _=done.send(result);
    });
    Ok((tx,result))
}
fn encode(encoder:&PathBuf,temp:&PathBuf,output:&PathBuf,rx:mpsc::Receiver<Message>)->Result<(),String> {
    let first=rx.recv().map_err(|e|e.to_string())?;
    let Message::Frame(_,image)=first else {return Err("No frames captured".into());};
    // Fixed even dimensions; resizing the app is fitted into this canvas.
    let width=(image.width as usize/2*2).min(1920);
    let height=((image.height as f64*width as f64/image.width as f64) as usize/2*2).max(2);
    let log=std::fs::File::create(temp.join("encoder.log")).map_err(|e|e.to_string())?;
    let mut child=Command::new(encoder).args(["-hide_banner","-loglevel","error","-y","-f","rawvideo","-pixel_format","rgba","-video_size",&format!("{width}x{height}"),"-framerate","30","-i","pipe:0","-an","-vf","vflip","-c:v","libx264","-preset","ultrafast","-crf","20","-pix_fmt","yuv420p"]).arg(temp.join("video.mp4")).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(log).spawn().map_err(|e|e.to_string())?;
    let mut stdin=child.stdin.take().unwrap();
    let mut last=fit(&image,width,height); let mut count=0;
    let write_result=(|| -> Result<(u64,Vec<Segment>),String> {
        stdin.write_all(&last).map_err(|e|e.to_string())?;count+=1;
        loop { match rx.recv().map_err(|_|"Recording interrupted".to_string())? {
            Message::Frame(index,img)=>{
                while count<index {stdin.write_all(&last).map_err(|e|e.to_string())?;count+=1;}
                last=fit(&img,width,height);stdin.write_all(&last).map_err(|e|e.to_string())?;count+=1;
            },
            Message::Finish(frames,segments)=>{
                while count<frames {stdin.write_all(&last).map_err(|e|e.to_string())?;count+=1;}
                return Ok((count,segments));
            }
        }}
    })();
    drop(stdin);let status=child.wait().map_err(|e|e.to_string())?;
    let (frames,segments)=write_result?;
    if !status.success(){return Err(format!("Video encoder failed; see {}",temp.display()));}
    let duration=frames as f64/FPS;
    std::fs::write(temp.join("music.mp3"),super::MUSIC_BYTES).map_err(|e|e.to_string())?;
    let mut cmd=Command::new(encoder);
    cmd.args(["-hide_banner","-loglevel","error","-y","-i"]).arg(temp.join("video.mp4"));
    cmd.args(["-f","lavfi","-i","anullsrc=r=44100:cl=stereo"]);
    for s in &segments {
        cmd.args(["-stream_loop","-1","-ss",&s.position.to_string(),"-t",&s.duration.to_string(),"-i"]).arg(temp.join("music.mp3"));
    }
    let mut filters=Vec::new();let mut inputs=String::from("[1:a]");
    for (i,s) in segments.iter().enumerate() {
        filters.push(format!("[{}:a]asetpts=PTS-STARTPTS,volume={},adelay={}:all=1[a{i}]",i+2,s.volume,(s.start*1000.0).round() as u64));
        inputs+=&format!("[a{i}]");
    }
    filters.push(format!("{inputs}amix=inputs={}:duration=first:normalize=0[a]",segments.len()+1));
    let result=cmd.args(["-filter_complex",&filters.join(";"),"-map","0:v","-map","[a]","-t",&duration.to_string(),"-c:v","copy","-c:a","aac","-b:a","192k","-movflags","+faststart"]).arg(output).output().map_err(|e|e.to_string())?;
    if !result.status.success(){return Err(String::from_utf8_lossy(&result.stderr).chars().take(200).collect());}
    Ok(())
}
fn fit(image:&Image,width:usize,height:usize)->Vec<u8> {
    let mut pixels=vec![0;width*height*4];
    let scale=(width as f64/image.width as f64).min(height as f64/image.height as f64);
    let w=(image.width as f64*scale) as usize;let h=(image.height as f64*scale) as usize;
    for y in 0..h {for x in 0..w {
        let src=(((y as f64/scale) as usize)*image.width as usize+(x as f64/scale) as usize)*4;
        let dst=((y+(height-h)/2)*width+x+(width-w)/2)*4;
        pixels[dst..dst+4].copy_from_slice(&image.bytes[src..src+4]);
    }}
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mp4_contains_video_and_only_scheduled_app_audio() {
        let encoder = ffmpeg().expect("FFmpeg required for recording integration test");
        let dir = std::env::temp_dir().join(format!("simroki-record-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("test.mp4");
        let (tx,rx) = mpsc::channel();
        tx.send(Message::Frame(0,Image { width: 64, height: 48, bytes: vec![128;64*48*4] })).unwrap();
        tx.send(Message::Finish(90,vec![
            Segment {start:0.5, position:40.0, duration:0.5, volume:0.5},
            Segment {start:1.5, position:40.5, duration:0.5, volume:0.5},
        ])).unwrap();
        encode(&encoder,&dir,&output,rx).unwrap();
        let audio=Command::new(&encoder).args(["-v","error","-i"]).arg(&output)
            .args(["-map","0:a:0","-f","f32le","-ac","1","-ar","8000","pipe:1"]).output().unwrap();
        assert!(audio.status.success());
        let samples:Vec<f32>=audio.stdout.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
        let energy=|a:f64,b:f64| samples[(a*8000.0) as usize..(b*8000.0) as usize].iter().map(|x|x*x).sum::<f32>();
        assert!(energy(0.1,0.4)<0.00001,"must start silent");
        assert!(energy(0.6,0.9)>0.01,"app music must be audible");
        assert!(energy(1.1,1.4)<0.00001,"paused music must be silent");
        assert!(energy(1.6,1.9)>0.01,"resumed music must be audible");
        assert!(energy(2.2,2.8)<0.00001,"music must stop at the recorded pause");
        let video=Command::new(&encoder).args(["-v","error","-i"]).arg(&output)
            .args(["-map","0:v:0","-f","null","-"]).output().unwrap();
        assert!(video.status.success());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
