use eframe::egui;
use eframe::egui::plot::{Bar, BarChart, Plot, Legend};
use eframe::epaint::Color32;
use lofty::mpeg::MpegFile;
use lofty::{AudioFile, ParseOptions};
use rodio::{Decoder, OutputStream};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::VecDeque;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const FFT_SIZE: usize = 1024;

struct App<T> {
    visual: Arc<Mutex<Visual<T>>>,
    duration: Duration,
    start: Instant,
    samples: Vec<i16>,
}

struct Visual<T> {
    samples: VecDeque<T>,
    window_size: usize,
}

impl<T> Visual<T> {
    fn new(window_size: usize) -> Self {
        Self {
            samples: VecDeque::<T>::new(),
            window_size,
        }
    }

    fn push(&mut self, value: T) {
        self.samples.push_back(value);
        while self.samples.len() > self.window_size {
            self.samples.pop_front();
        }
    }
}

impl<T> App<T> {
    fn new(window_size: usize, duration: Duration, start: Instant, samples: Vec<i16>) -> Self {
        Self {
            visual: Arc::new(Mutex::new(Visual::new(window_size))),
            duration,
            start,
            samples,
        }
    }
}

impl<T> eframe::App for App<T>
where
    T: Into<f64> + std::convert::From<f32> + Copy,
{
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Audio Visualization");
            let points = self
                .visual
                .lock()
                .unwrap()
                .samples
                .iter()
                .enumerate()
                .map(|(i, val)| Bar::new(i as f64, (*val).into()))
                .collect();
            let chart = BarChart::new(points).color(Color32::LIGHT_BLUE);
            Plot::new("visual")
                .legend(Legend::default())
                .allow_zoom(false)
                .allow_drag(false)
                .show_x(true)
                .show_y(true)
                .show(ui, |plot_ui| plot_ui.bar_chart(chart));
        });

        // 当前位置
        let pos = self.start.elapsed().as_secs_f64() / self.duration.as_secs_f64();
        let start_index: usize = (self.samples.len() as f64 * pos).floor() as usize;
        let end_index: usize = start_index + FFT_SIZE;
        if end_index < self.samples.len() {
            let samples = &self.samples[start_index..start_index + FFT_SIZE];
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(FFT_SIZE);
            let mut buffer: Vec<Complex<f32>> = samples
                .iter()
                .map(|real| Complex {
                    re: *real as f32,
                    im: 0.0,
                })
                .collect();
            fft.process(&mut buffer);
            buffer
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .for_each(|y| self.visual.lock().unwrap().push(y.into()));
        }

        ctx.request_repaint();
    }
}

fn main() -> Result<(), eframe::Error> {
    let path = env::args().nth(1).unwrap();
    let mut file_content = File::open(&path).unwrap();
    let audio_file = MpegFile::read_from(&mut file_content, ParseOptions::new()).unwrap();
    let duration = audio_file.properties().duration();
    let start = Instant::now();

    let (_stream, _stream_handle) = OutputStream::try_default().unwrap();
    let file = BufReader::new(File::open(path).unwrap());
    let source = Decoder::new(file).unwrap();

    let app: App<f32> = App::new(FFT_SIZE, duration, start, source.collect());
    let options = eframe::NativeOptions::default();
    eframe::run_native("Audio Visualaztion", options, Box::new(|_cc| Box::new(app)))
}
