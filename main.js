// Import necessary modules from Electron and Node.js
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http'); // Import the http module for pinging

let pythonProcess = null;
let mainWindow = null;

// --- The Pinging Function ---
// This function will repeatedly try to connect to the Gradio server.
const checkGradioReady = (url, callback) => {
  const tryConnect = () => {
    http.get(url, (res) => {
      // If we get a response (any response), the server is up!
      if (res.statusCode >= 200 && res.statusCode < 400) {
        console.log('Gradio server is ready!');
        callback();
      } else {
        setTimeout(tryConnect, 1000); // Server is up but not ready, try again
      }
    }).on('error', () => {
      // If the connection is refused, the server is not up yet.
      console.log('Gradio not ready yet, retrying...');
      setTimeout(tryConnect, 1000); // Retry every second
    });
  };
  tryConnect();
};


function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    title: 'Glimpse',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.loadURL(`data:text/html;charset=utf-8, <h1>Loading Glimpse AI Engine...</h1><p>This may take a moment. Please be patient.</p>`);

  console.log('Starting Python backend server...');
  
  const pythonExecutable = process.platform === 'win32'
    ? path.join(__dirname, 'backend', 'venv', 'Scripts', 'python.exe')
    : path.join(__dirname, 'backend', 'venv', 'bin', 'python');
  
  const scriptPath = path.join(__dirname, 'backend', 'app.py');

  pythonProcess = spawn(pythonExecutable, ['-u', scriptPath], {
    cwd: path.join(__dirname, 'backend')
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python stdout: ${data}`);
  });
  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data}`);
  });

  // --- Use the new, robust checking method ---
  checkGradioReady('http://127.0.0.1:7860', () => {
    if (mainWindow) {
      mainWindow.loadURL('http://127.0.0.1:7860');
    }
  });
}

// Standard Electron boilerplate...
app.whenReady().then(createMainWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  if (pythonProcess) {
    console.log('Terminating Python backend process...');
    pythonProcess.kill();
    pythonProcess = null;
  }
});

