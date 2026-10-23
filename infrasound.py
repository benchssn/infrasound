#!/usr/bin/env python3
"""Start with: python infrasound.py [recording.wav]"""
import sys
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy import signal
import scipy
import soundfile as sf
import sounddevice as sd
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters
from processing import Settings, load_recording, isolate, accelerated, spectrum, summary, response

pg.setConfigOptions(antialias=True, background='w', foreground='k')

def read_source(path):
    # Hash and decode the same open file, before processing.
    with open(path, 'rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
        f.seek(0)
        data, sr = load_recording(f)
    return data, sr, path, h.hexdigest()

class Worker(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()
    def __init__(self, fn):
        super().__init__(); self.fn = fn
    @QtCore.Slot()
    def run(self):
        try: self.done.emit(self.fn())
        except Exception as e: self.failed.emit(str(e))
        finally: self.finished.emit()

class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Infrasound analysis'); self.resize(1250, 850)
        self.recording = None; self.result = None; self.path = None; self.busy = False
        self.audio = None; self.audio_position = 0; self.playing = False
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        controls = QtWidgets.QWidget(); controls.setMaximumWidth(340)
        form = QtWidgets.QVBoxLayout(controls)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls); scroll.setMinimumWidth(335); scroll.setMaximumWidth(360); layout.addWidget(scroll)
        openrow = QtWidgets.QHBoxLayout(); self.open_btn = QtWidgets.QPushButton('Open WAV / FLAC…'); self.demo_btn = QtWidgets.QPushButton('Demo')
        openrow.addWidget(self.open_btn); openrow.addWidget(self.demo_btn); form.addLayout(openrow)
        self.file_label = QtWidgets.QLabel('No recording loaded'); self.file_label.setWordWrap(True); form.addWidget(self.file_label)
        grid = QtWidgets.QFormLayout(); form.addLayout(grid)
        self.channel = QtWidgets.QComboBox(); grid.addRow('Channel', self.channel)
        self.start = self.number(0, 864000, 0, 3); self.end = self.number(0, 864000, 30, 3)
        grid.addRow('Start (original seconds)', self.start); grid.addRow('End (original seconds)', self.end)
        self.low = self.number(.01, 19.99, .5, 2); self.high = self.number(.02, 20, 20, 2)
        grid.addRow('Lower cutoff (Hz)', self.low); grid.addRow('Upper cutoff (Hz)', self.high)
        self.order = QtWidgets.QSpinBox(); self.order.setRange(1, 12); self.order.setValue(8); grid.addRow('Order per filter edge', self.order)
        self.phase = QtWidgets.QComboBox(); self.phase.addItems(['zero-phase', 'causal']); grid.addRow('Offline filter mode', self.phase)
        self.speed = self.number(1, 100, 20, 1); grid.addRow('Acceleration (×)', self.speed)
        self.normalize = QtWidgets.QCheckBox('Normalize listening / accelerated export'); form.addWidget(self.normalize)
        self.process_btn = QtWidgets.QPushButton('Analyze selection'); form.addWidget(self.process_btn)
        self.stats = QtWidgets.QLabel('Load a recording or generate a demo.'); self.stats.setWordWrap(True); self.stats.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse); form.addWidget(self.stats)
        self.notes = QtWidgets.QLabel('Amplitudes are uncalibrated digital full scale (FS), not sound pressure. Your sensor must capture frequencies below 20 Hz.'); self.notes.setWordWrap(True); form.addWidget(self.notes)
        self.play_source = QtWidgets.QComboBox(); self.play_source.addItems(['Isolated (accelerated)', 'Original (accelerated)']); form.addWidget(self.play_source)
        row = QtWidgets.QHBoxLayout(); self.play_btn = QtWidgets.QPushButton('Play'); self.stop_btn = QtWidgets.QPushButton('Stop'); row.addWidget(self.play_btn); row.addWidget(self.stop_btn); form.addLayout(row)
        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel('Volume')); self.volume = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.volume.setRange(0,100); self.volume.setValue(25); row.addWidget(self.volume); form.addLayout(row)
        self.loop = QtWidgets.QCheckBox('Loop'); form.addWidget(self.loop)
        self.seek = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.seek.setRange(0,1000); form.addWidget(self.seek)
        self.clock = QtWidgets.QLabel('0.00 s / 0.00 s (listening time)'); form.addWidget(self.clock)
        self.export_btn = QtWidgets.QPushButton('Export audio + processing record…'); self.plot_btn = QtWidgets.QPushButton('Export current plot as PNG…'); form.addWidget(self.export_btn); form.addWidget(self.plot_btn); form.addStretch()
        self.tabs = QtWidgets.QTabWidget(); layout.addWidget(self.tabs, 1)
        self.wave = self.plot('Waveform', 'Amplitude (FS)', 'Original time', 's')
        self.psd = self.plot('Power spectral density', 'PSD (dB re 1 FS²/Hz)', 'Original frequency', 'Hz')
        self.spectro = self.plot('Spectrogram', 'Original frequency (Hz)', 'Original time', 's')
        self.filter_plot = self.plot('Filter response', 'Magnitude (dB)', 'Original frequency', 'Hz'); self.filter_plot.setLogMode(x=True)
        self.timer = QtCore.QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.tick); self.timer.start()
        self.open_btn.clicked.connect(self.open_file); self.demo_btn.clicked.connect(self.demo); self.process_btn.clicked.connect(self.analyze)
        self.play_btn.clicked.connect(self.play); self.stop_btn.clicked.connect(self.stop); self.seek.sliderReleased.connect(self.seek_audio)
        self.export_btn.clicked.connect(self.export); self.plot_btn.clicked.connect(self.export_plot)
        for widget in [self.channel, self.phase, self.play_source]: widget.currentIndexChanged.connect(self.invalidate)
        for widget in [self.start,self.end,self.low,self.high,self.order,self.speed]: widget.valueChanged.connect(self.invalidate)
        self.normalize.toggled.connect(self.invalidate)
        self.set_ready(False)

    @staticmethod
    def number(low, high, value, decimals):
        w=QtWidgets.QDoubleSpinBox(); w.setRange(low,high); w.setDecimals(decimals); w.setValue(value); return w
    def plot(self,title,y,x,unit):
        p=pg.PlotWidget(); p.setLabel('left',y); p.setLabel('bottom',x,units=unit); p.showGrid(x=True,y=True,alpha=.2); p.addLegend(); self.tabs.addTab(p,title); return p
    def set_ready(self,ready):
        for w in [self.play_btn,self.export_btn,self.plot_btn]: w.setEnabled(ready and not self.busy)
    def invalidate(self,*args):
        self.stop(); self.audio=None; self.result=None; self.set_ready(False)
        if self.recording is not None: self.statusBar().showMessage('Settings changed. Analyze selection to refresh results.')
    def run_job(self, fn, complete):
        if self.busy: return
        self.busy=True; self.set_ready(False)
        self.centralWidget().setEnabled(False); self.statusBar().showMessage('Processing…')
        self.thread=QtCore.QThread(self); self.worker=Worker(fn); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run); self.worker.done.connect(complete); self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.thread.quit); self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.job_finished); self.thread.finished.connect(self.thread.deleteLater); self.thread.start()
    def job_finished(self):
        self.busy=False; self.centralWidget().setEnabled(True); self.set_ready(self.result is not None)
    def error(self,message):
        self.statusBar().showMessage('Operation failed'); QtWidgets.QMessageBox.warning(self,'Infrasound',message)
    def open_file(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Open recording','','Audio (*.wav *.flac *.aiff *.aif *.ogg);;All files (*)')
        if path:
            self.stop(); self.run_job(lambda:read_source(path), self.loaded)
    def loaded(self, result):
        self.recording,self.sr,self.path,self.source_hash=result; self.result=None; self.audio=None
        self.channel.blockSignals(True); self.channel.clear()
        self.channel.addItems([f'Channel {i+1}' for i in range(self.recording.shape[1])]); self.channel.addItem('Mean of channels'); self.channel.blockSignals(False)
        duration=len(self.recording)/self.sr
        self.start.setMaximum(duration); self.end.setMaximum(duration); self.start.setValue(0); self.end.setValue(duration)
        self.file_label.setText(f'{Path(self.path).name}\n{self.sr:,} Hz · {self.recording.shape[1]} channel(s) · {duration:.3f} s')
        self.stats.setText('Recording loaded. Choose settings and click Analyze selection.'); self.statusBar().showMessage('Ready to analyze')
    def demo(self):
        sr=2000; t=np.arange(sr*60)/sr
        x=.18*np.sin(2*np.pi*8*t)+.08*np.sin(2*np.pi*14*t)+.12*np.sin(2*np.pi*40*t)
        self.loaded((x[:,None],sr,'Generated demo (8, 14, 40 Hz)',None))
    def analyze(self):
        if self.recording is None: return self.error('Open a recording first.')
        self.stop()
        a=round(self.start.value()*self.sr); b=round(self.end.value()*self.sr)
        if b<=a: return self.error('End must be after start.')
        settings=Settings(self.low.value(),self.high.value(),self.order.value(),self.phase.currentText(),self.speed.value(),self.normalize.isChecked())
        channel=self.channel.currentIndex(); x=self.recording[a:b]
        x=x[:,channel].copy() if channel<x.shape[1] else x.mean(axis=1)
        # Capture all UI values before starting the worker.
        channel_name=self.channel.currentText()
        def safe_job():
            y=isolate(x,self.sr,settings); heard,rate=accelerated(y,self.sr,settings.speed,settings.normalize)
            original,_=accelerated(x,self.sr,settings.speed,False)
            return dict(x=x,y=y,heard=heard,original=original,rate=rate,settings=settings,start=a/self.sr,end=b/self.sr,channel=channel_name,stats=summary(y,self.sr,settings))
        self.run_job(safe_job,self.analyzed)
    def analyzed(self,r):
        try:
            self.render_result(r)
        except Exception as e:
            self.result=None; self.set_ready(False); self.error(f'Could not display analysis: {e}')
    def render_result(self,r):
        self.result=r; s=r['stats']; settings=r['settings']; peak=s['dominant_hz']
        self.stats.setText(f"Dominant frequency: {peak:.3f} Hz\n" if peak is not None else 'Dominant frequency: none\n')
        self.stats.setText(self.stats.text()+f"RMS: {s['rms_fs']:.6g} FS ({s['rms_dbfs']:.2f} dBFS)\nPeak: {s['peak_fs']:.6g} FS\nPSD bin spacing: {s['psd_bin_spacing_hz']:.4g} Hz\nListening duration: {s['audible_duration_seconds']:.3f} s\nCutoff mapping: {settings.low_hz*settings.speed:g}–{settings.high_hz*settings.speed:g} Hz")
        edge=min((r['end']-r['start'])/2,3/settings.low_hz)
        self.notes.setText(f'Uncalibrated FS, not SPL. Shaded edges ({edge:.2f} s each) are a heuristic transient caution, not a guaranteed settling bound. '+('Forward/backward filtering doubles effective order and squares the magnitude response.' if settings.phase=='zero-phase' else 'Causal filtering introduces phase delay. Preview uses this offline filtered result.'))
        self.wave.clear(); self.psd.clear(); self.spectro.clear(); self.filter_plot.clear()
        for key,color,name in [('x',(130,130,130),'Original'),('y',(0,90,180),'Isolated')]:
            d=r[key]; time=r['start']+np.arange(len(d))/self.sr
            self.wave.plot(time,d,pen=color,name=name,autoDownsample=True,downsampleMethod='peak')
            f,p=spectrum(d,self.sr); keep=f<=max(50,settings.high_hz*2)
            self.psd.plot(f[keep],10*np.log10(np.maximum(p[keep],1e-20)),pen=color,name=name)
        for bounds in [(r['start'],r['start']+edge),(r['end']-edge,r['end'])]:
            self.wave.addItem(pg.LinearRegionItem(bounds,movable=False,brush=(220,140,0,30),pen=pg.mkPen(None)))
        n=min(len(r['y']),max(128,int(self.sr*8)))
        f,t,p=signal.spectrogram(r['y'],self.sr,window='hann',nperseg=n,noverlap=n*3//4,scaling='density',mode='psd')
        keep=f<=min(50,self.sr/2); f=f[keep]; p=p[keep]
        image=pg.ImageItem(axisOrder='row-major'); image.setImage(10*np.log10(np.maximum(p,1e-20)),levels=(-100,-10)); image.setColorMap(pg.colormap.get('viridis'))
        dt=(t[1]-t[0]) if len(t)>1 else n/self.sr; df=f[1]-f[0]
        image.setRect(QtCore.QRectF(r['start']+t[0]-dt/2,-df/2,dt*len(t),df*len(f))); self.spectro.addItem(image)
        self.spectro.setTitle('Isolated signal · PSD color scale −100 to −10 dB re 1 FS²/Hz')
        f,h=response(self.sr,settings); self.filter_plot.plot(f,h,pen=(0,90,180),name=settings.phase); self.filter_plot.setYRange(-100,5)
        self.wave.setXRange(r['start'],r['end']); self.psd.setXRange(0,min(50,self.sr/2)); self.spectro.setYRange(0,min(50,self.sr/2))
        self.statusBar().showMessage('Analysis complete. Metrics use the filtered signal before normalization.'); self.set_ready(True)
    def callback(self,outdata,frames,time,status):
        outdata.fill(0); pos=0
        while pos<frames:
            if self.audio_position>=len(self.audio):
                if self.looping: self.audio_position=0
                else: raise sd.CallbackStop
            count=min(frames-pos,len(self.audio)-self.audio_position)
            outdata[pos:pos+count,0]=np.clip(self.audio[self.audio_position:self.audio_position+count]*self.play_volume,-1,1)
            pos+=count; self.audio_position+=count
    def play(self):
        if self.result is None: return
        if self.playing: self.stop(reset=False); return
        self.audio=self.result['heard' if self.play_source.currentIndex()==0 else 'original']
        self.play_volume=self.volume.value()/100; self.looping=self.loop.isChecked()
        try:
            self.stream=sd.OutputStream(samplerate=self.result['rate'],channels=1,callback=self.callback); self.stream.start(); self.playing=True; self.play_btn.setText('Pause')
        except Exception as e: self.stop(); self.error(f'Audio output unavailable: {e}')
    def stop(self,checked=False,reset=True):
        if hasattr(self,'stream'):
            self.stream.abort(); self.stream.close(); del self.stream
        self.playing=False; self.play_btn.setText('Play')
        if reset: self.audio_position=0; self.seek.setValue(0)
    def tick(self):
        self.play_volume=self.volume.value()/100; self.looping=self.loop.isChecked()
        if self.result and self.audio is not None:
            length=len(self.audio)
            if not self.seek.isSliderDown(): self.seek.setValue(round(1000*self.audio_position/max(1,length)))
            self.clock.setText(f'{self.audio_position/48000:.2f} s / {length/48000:.2f} s (listening time)')
        if self.playing and hasattr(self,'stream') and not self.stream.active: self.stop()
    def seek_audio(self):
        if self.result is None: return
        was=self.playing; self.stop(reset=False)
        self.audio=self.result['heard' if self.play_source.currentIndex()==0 else 'original']; self.audio_position=min(len(self.audio)-1,round(self.seek.value()/1000*len(self.audio)))
        if was:self.play()
    def export(self):
        if self.result is None:return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Export accelerated WAV (also saves filtered WAV and JSON)','','WAV (*.wav)')
        if not path:return
        p=Path(path).with_suffix('.wav'); filtered=p.with_name(p.stem+'_filtered.wav'); metadata=p.with_suffix('.json')
        if any(q.exists() for q in [filtered,metadata]):
            if QtWidgets.QMessageBox.question(self,'Replace files?','A companion filtered WAV or JSON already exists. Replace the export files?')!=QtWidgets.QMessageBox.Yes:return
        r=self.result
        def job():
            sf.write(p,r['heard'],r['rate'],subtype='FLOAT'); sf.write(filtered,r['y'],self.sr,subtype='FLOAT')
            digest=self.source_hash
            record=dict(source=str(self.path),source_sha256=digest,sample_rate=self.sr,channel=r['channel'],selection_seconds=[r['start'],r['end']],settings=asdict(r['settings']),measurements=r['stats'],amplitude_units='uncalibrated FS; PSD in FS^2/Hz',filter='Butterworth highpass then lowpass; order per edge; zero-phase squares magnitude response',versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,soundfile=sf.__version__),outputs=dict(accelerated=str(p),filtered=str(filtered)),normalization='Applied only to accelerated output if selected; filtered WAV and measurements retain original scale.')
            metadata.write_text(json.dumps(record,indent=2,allow_nan=False)); return str(p)
        self.run_job(job,lambda p:self.statusBar().showMessage(f'Exported {p}, filtered WAV, and JSON record'))
    def export_plot(self):
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Export plot','','PNG (*.png)')
        if path:
            try: pg.exporters.ImageExporter(self.tabs.currentWidget().plotItem).export(str(Path(path).with_suffix('.png')))
            except Exception as e:self.error(str(e))
    def closeEvent(self,event):
        if self.busy:
            event.ignore(); self.statusBar().showMessage('Wait for processing to finish before closing.'); return
        self.stop(); event.accept()

if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv); window=Window(); window.show()
    if len(sys.argv)>1:
        path=sys.argv[1]; window.run_job(lambda:read_source(path),window.loaded)
    sys.exit(app.exec())
