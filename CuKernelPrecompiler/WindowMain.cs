
namespace CuKernelPrecompiler
{
	public partial class WindowMain : Form
	{
		// ----- ----- ----- ATTRIBUTES ----- ----- ----- //
		public string Repopath;

		public AudioHandling AudioH;
		public CudaHandling CudaH;



		private int _oldChunkSize = 65536;



		// ----- ----- ----- LAMBDA FUNCTIONS ----- ----- ----- //
		public AudioObject? SelectedTrack => AudioH.CurrentTrack;

		public bool DataOnHost => AudioH.CurrentTrack?.Floats.Length > 0 && AudioH.CurrentTrack?.Pointer == 0;
		public bool DataOnCuda => AudioH.CurrentTrack?.Pointer != 0 && AudioH.CurrentTrack?.Floats.Length == 0;

		public bool DataTransformed => AudioH.CurrentTrack?.Floats.Length == 0 && AudioH.CurrentTrack?.Pointer != 0 && CudaH.DeviceComplexVars.ContainsKey(AudioH.CurrentTrack?.Pointer ?? 0);

		public bool KernelLoaded => CudaH.Kernel != null;


		// ----- ----- ----- CONSTRUCTORS ----- ----- ----- //
		public WindowMain()
		{
			InitializeComponent();
			Repopath = GetRepopath(true);

			// Window position
			this.StartPosition = FormStartPosition.Manual;
			this.Location = new Point(0, 0);

			// Init. classes
			AudioH = new AudioHandling(Repopath, listBox_tracks, pictureBox_waveform, button_playback, textBox_timestamp, button_colorGraph, button_colorBackground, label_trackMeta, hScrollBar_offset);
			CudaH = new CudaHandling(Repopath, listBox_log, comboBox_cudaDevice, label_vram, progressBar_vram, listBox_kernels, label_kernelLoaded);


			// Register events
			listBox_tracks.SelectedIndexChanged += (sender, e) => ToggleUI();
			textBox_kernelString.TextChanged += (sender, e) => ToggleUI();
			pictureBox_waveform.Click += (sender, e) => ExportWav();


			// Init. audio files
			ImportResourcesAudios();
		}







		// ----- ----- ----- METHODS ----- ----- ----- //
		private string GetRepopath(bool root)
		{
			string repo = AppDomain.CurrentDomain.BaseDirectory;

			if (root)
			{
				repo += @"..\..\..\";
			}

			repo = Path.GetFullPath(repo);
			return repo;
		}

		public void ImportResourcesAudios()
		{
			// Get each wav mp3 flac file in Resources/Audios
			string[] files = Directory.GetFiles(Repopath + @"Resources\Audios\", "*.*", SearchOption.AllDirectories);

			// Add each file with AudioH
			foreach (string file in files)
			{
				AudioH.AddTrack(file);
			}
		}

		public void ToggleUI()
		{
			// Textbox scroll at 25+ lines
			textBox_kernelString.ScrollBars = textBox_kernelString.Lines.Length > 25 ? ScrollBars.Vertical : ScrollBars.None;

			// Button playback
			button_playback.Enabled = SelectedTrack != null && DataOnHost;

			// Button move
			button_cudaMove.Enabled = SelectedTrack != null && (DataOnHost || DataOnCuda) && !DataTransformed;
			button_cudaMove.Text = DataOnHost ? "-> CUDA" : "Host <-";

			// Button transform
			button_cudaTransform.Enabled = SelectedTrack != null && DataOnCuda;
			button_cudaTransform.Text = DataTransformed ? "I-FFT" : "FFT";

			// Button normalize
			button_normalize.Enabled = SelectedTrack != null && DataOnHost;

			// Button kernel run
			button_kernelRun.Enabled = SelectedTrack != null && DataOnCuda && KernelLoaded;

		}

		public void ExportWav()
		{
			// Abort if not CTRL down
			if (Control.ModifierKeys != Keys.Control)
			{
				return;
			}

			// OFD at MyMusic
			SaveFileDialog sfd = new SaveFileDialog();
			sfd.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);
			sfd.Filter = "WAV files (*.wav)|*.wav";
			sfd.FileName = SelectedTrack?.Name ?? "track";

			// Export
			if (sfd.ShowDialog() == DialogResult.OK)
			{
				SelectedTrack?.ExportAudioWav(sfd.FileName);
			}

			// MsgBox
			MessageBox.Show("Exported to " + sfd.FileName);
		}



		// ----- ----- ----- EVENTS ----- ----- ----- //
		private void numericUpDown_chunkSize_ValueChanged(object sender, EventArgs e)
		{
			// Double or half the chunk size if increasing or decreasing
			if (numericUpDown_chunkSize.Value > _oldChunkSize)
			{
				numericUpDown_chunkSize.Value = Math.Min(numericUpDown_chunkSize.Maximum, _oldChunkSize * 2);
			}
			else if (numericUpDown_chunkSize.Value < _oldChunkSize)
			{
				numericUpDown_chunkSize.Value = Math.Max(numericUpDown_chunkSize.Minimum, _oldChunkSize / 2);
			}

			// Update chunk size
			_oldChunkSize = (int) numericUpDown_chunkSize.Value;
		}

		private void button_cudaMove_Click(object sender, EventArgs e)
		{
			if (SelectedTrack == null)
			{
				return;
			}

			// Move data to/from CUDA
			if (DataOnHost)
			{
				SelectedTrack.Pointer = CudaH.PushChunks(SelectedTrack.MakeChunks((int) numericUpDown_chunkSize.Value, (int) numericUpDown_chunkSize.Value / 2) ?? []);

				if (SelectedTrack.Pointer != 0)
				{
					SelectedTrack.Floats = [];
				}
			}
			else if (DataOnCuda)
			{
				SelectedTrack.AggregateChunks(CudaH.PullChunks<float>(SelectedTrack.Pointer));

				if (SelectedTrack.Floats.Length > 0)
				{
					SelectedTrack.Pointer = 0;
				}
			}

			// Update UI
			ToggleUI();
		}

		private void button_cudaTransform_Click(object sender, EventArgs e)
		{
			if (SelectedTrack == null || !DataOnCuda)
			{
				return;
			}

			// Transform data on CUDA
			if (!DataTransformed)
			{
				SelectedTrack.Pointer = CudaH.PerformFFT(SelectedTrack.Pointer);
			}
			else
			{
				SelectedTrack.Pointer = CudaH.PerformIFFT(SelectedTrack.Pointer);
			}

			// Update UI
			ToggleUI();
		}

		private void button_normalize_Click(object sender, EventArgs e)
		{
			if (SelectedTrack == null)
			{
				return;
			}

			// Normalize data
			SelectedTrack.Normalize();

			// Update UI
			ToggleUI();
		}

		private void comboBox_cudaDevice_SelectedIndexChanged(object sender, EventArgs e)
		{
			// Toggle UI
			ToggleUI();
		}

		private void button_kernelRun_Click(object sender, EventArgs e)
		{
			// Abort if no track or no data
			if (SelectedTrack == null || SelectedTrack.Pointer == 0)
			{
				return;
			}

			// Load kernel
			CudaH.LoadKernelByName(listBox_kernels.SelectedItem?.ToString() ?? "");

			// Run kernel if param2 disabled (single float param)
			if (numericUpDown_param2.Enabled == false)
			{
				CudaH.RunKernel(SelectedTrack.Pointer, (float) numericUpDown_param1.Value);
			}
			else
			{
				CudaH.RunKernel(SelectedTrack.Pointer, (float) numericUpDown_param1.Value, (float) numericUpDown_param2.Value);
			}

			// Update UI
			ToggleUI();
		}

		private void button_kernelLoad_Click(object sender, EventArgs e)
		{
			// Load kernel
			CudaH.LoadKernelByName(listBox_kernels.SelectedItem?.ToString() ?? "");

			// Update UI
			ToggleUI();

		}

		private void button_kernelCompile_Click(object sender, EventArgs e)
		{
			// Compile kernel
			string ptxpath = CudaH.CompileString(textBox_kernelString.Text);

			// Load kernel in list
			CudaH.VerifyCompiledKernels(listBox_kernels, true);

			// Load kernel
			CudaH.LoadKernel(ptxpath);

			// Update UI
			ToggleUI();
		}
	}
}
