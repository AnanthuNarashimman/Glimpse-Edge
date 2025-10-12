// Import necessary modules from Electron and Node.js
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let pythonProcess = null;

function createMainWindow() {
  console.log('Starting Python backend server...');

  // --- FIX #1: Define the correct path to the Python executable ---
  // This explicitly uses the Python from your virtual environment.
  const pythonExecutable = process.platform === 'win32'
    ? path.join(__dirname, 'backend', 'venv', 'Scripts', 'python.exe')
    : path.join(__dirname, 'backend', 'venv', 'bin', 'python');
  
  const scriptPath = path.join(__dirname, 'backend', 'app.py');

  // --- FIX #2: Set the correct working directory for the Python script ---
  // The 'cwd' option tells the script to run *from within* the 'backend' folder.
  // This is crucial for it to find the model file and other relative paths.
  pythonProcess = spawn(pythonExecutable, [scriptPath], {
    cwd: path.join(__dirname, 'backend') 
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python stdout: ${data}`);
  });
  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data}`);
  });
  
  pythonProcess.on('error', (error) => {
    console.error(`Failed to start Python process: ${error.message}`);
  });
  
  pythonProcess.on('exit', (code, signal) => {
    console.log(`Python process exited with code ${code} and signal ${signal}`);
    if (code !== 0 && code !== null) {
      console.error('⚠️ Python backend crashed! Check the error messages above.');
    }
  });

  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    title: 'Glimpse',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  const gradioUrl = 'http://127.0.0.1:7860';

  const loadUrlWithRetries = () => {
    mainWindow.loadURL(gradioUrl)
      .then(() => {
        console.log('✅ Connection successful! Gradio UI loaded.');
      })
      .catch(() => {
        console.log('⏳ Connection failed, retrying in 2 seconds...');
        setTimeout(loadUrlWithRetries, 2000);
      });
  };
  
  setTimeout(loadUrlWithRetries, 5000);
}

app.whenReady().then(createMainWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createMainWindow();
  }
});

app.on('will-quit', () => {
  if (pythonProcess) {
    console.log('Terminating Python backend process...');
    pythonProcess.kill();
    pythonProcess = null;
  }
});

