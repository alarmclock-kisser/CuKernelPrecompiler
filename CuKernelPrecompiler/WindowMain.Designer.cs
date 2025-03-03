namespace CuKernelPrecompiler
{
    partial class WindowMain
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

		#region Windows Form Designer generated code

		/// <summary>
		///  Required method for Designer support - do not modify
		///  the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			listBox_log = new ListBox();
			listBox_tracks = new ListBox();
			pictureBox_waveform = new PictureBox();
			groupBox_controls = new GroupBox();
			button_normalize = new Button();
			button_colorBackground = new Button();
			button_colorGraph = new Button();
			textBox_timestamp = new TextBox();
			button_playback = new Button();
			label_trackMeta = new Label();
			hScrollBar_offset = new HScrollBar();
			comboBox_cudaDevice = new ComboBox();
			label_vram = new Label();
			progressBar_vram = new ProgressBar();
			listBox_kernels = new ListBox();
			label_kernelLoaded = new Label();
			groupBox_cudaMemory = new GroupBox();
			label_chunkSize = new Label();
			numericUpDown_chunkSize = new NumericUpDown();
			button_cudaMove = new Button();
			groupBox_cudaFft = new GroupBox();
			button_cudaTransform = new Button();
			groupBox_cudaKernel = new GroupBox();
			button_kernelRun = new Button();
			label_kernelParam2 = new Label();
			numericUpDown_param2 = new NumericUpDown();
			label_kernelParam1 = new Label();
			numericUpDown_param1 = new NumericUpDown();
			button_kernelLoad = new Button();
			button_kernelCompile = new Button();
			textBox_kernelString = new TextBox();
			((System.ComponentModel.ISupportInitialize) pictureBox_waveform).BeginInit();
			groupBox_controls.SuspendLayout();
			groupBox_cudaMemory.SuspendLayout();
			((System.ComponentModel.ISupportInitialize) numericUpDown_chunkSize).BeginInit();
			groupBox_cudaFft.SuspendLayout();
			groupBox_cudaKernel.SuspendLayout();
			((System.ComponentModel.ISupportInitialize) numericUpDown_param2).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_param1).BeginInit();
			SuspendLayout();
			// 
			// listBox_log
			// 
			listBox_log.Font = new Font("Bahnschrift Light SemiCondensed", 8.25F, FontStyle.Regular, GraphicsUnit.Point,  0);
			listBox_log.FormattingEnabled = true;
			listBox_log.ItemHeight = 13;
			listBox_log.Location = new Point(12, 800);
			listBox_log.Name = "listBox_log";
			listBox_log.Size = new Size(614, 108);
			listBox_log.TabIndex = 0;
			// 
			// listBox_tracks
			// 
			listBox_tracks.FormattingEnabled = true;
			listBox_tracks.ItemHeight = 15;
			listBox_tracks.Location = new Point(632, 800);
			listBox_tracks.Name = "listBox_tracks";
			listBox_tracks.Size = new Size(140, 109);
			listBox_tracks.TabIndex = 1;
			// 
			// pictureBox_waveform
			// 
			pictureBox_waveform.BackColor = Color.White;
			pictureBox_waveform.Location = new Point(12, 679);
			pictureBox_waveform.Name = "pictureBox_waveform";
			pictureBox_waveform.Size = new Size(614, 100);
			pictureBox_waveform.TabIndex = 2;
			pictureBox_waveform.TabStop = false;
			// 
			// groupBox_controls
			// 
			groupBox_controls.Controls.Add(button_normalize);
			groupBox_controls.Controls.Add(button_colorBackground);
			groupBox_controls.Controls.Add(button_colorGraph);
			groupBox_controls.Controls.Add(textBox_timestamp);
			groupBox_controls.Controls.Add(button_playback);
			groupBox_controls.Location = new Point(632, 695);
			groupBox_controls.Name = "groupBox_controls";
			groupBox_controls.Size = new Size(140, 84);
			groupBox_controls.TabIndex = 3;
			groupBox_controls.TabStop = false;
			groupBox_controls.Text = "Controls";
			// 
			// button_normalize
			// 
			button_normalize.Location = new Point(6, 22);
			button_normalize.Name = "button_normalize";
			button_normalize.Size = new Size(70, 23);
			button_normalize.TabIndex = 13;
			button_normalize.Text = "Normalize";
			button_normalize.UseVisualStyleBackColor = true;
			button_normalize.Click += button_normalize_Click;
			// 
			// button_colorBackground
			// 
			button_colorBackground.BackColor = Color.White;
			button_colorBackground.ForeColor = Color.Black;
			button_colorBackground.Location = new Point(111, 22);
			button_colorBackground.Name = "button_colorBackground";
			button_colorBackground.Size = new Size(23, 23);
			button_colorBackground.TabIndex = 6;
			button_colorBackground.Text = "B";
			button_colorBackground.UseVisualStyleBackColor = false;
			// 
			// button_colorGraph
			// 
			button_colorGraph.BackColor = SystemColors.HotTrack;
			button_colorGraph.ForeColor = Color.White;
			button_colorGraph.Location = new Point(82, 22);
			button_colorGraph.Name = "button_colorGraph";
			button_colorGraph.Size = new Size(23, 23);
			button_colorGraph.TabIndex = 5;
			button_colorGraph.Text = "G";
			button_colorGraph.UseVisualStyleBackColor = false;
			// 
			// textBox_timestamp
			// 
			textBox_timestamp.Location = new Point(35, 55);
			textBox_timestamp.Name = "textBox_timestamp";
			textBox_timestamp.Size = new Size(99, 23);
			textBox_timestamp.TabIndex = 4;
			// 
			// button_playback
			// 
			button_playback.Location = new Point(6, 55);
			button_playback.Name = "button_playback";
			button_playback.Size = new Size(23, 23);
			button_playback.TabIndex = 4;
			button_playback.Text = ">";
			button_playback.UseVisualStyleBackColor = true;
			// 
			// label_trackMeta
			// 
			label_trackMeta.AutoSize = true;
			label_trackMeta.Font = new Font("Segoe UI", 8.25F, FontStyle.Regular, GraphicsUnit.Point,  0);
			label_trackMeta.Location = new Point(632, 679);
			label_trackMeta.Name = "label_trackMeta";
			label_trackMeta.Size = new Size(89, 13);
			label_trackMeta.TabIndex = 4;
			label_trackMeta.Text = "No track loaded";
			// 
			// hScrollBar_offset
			// 
			hScrollBar_offset.Location = new Point(12, 782);
			hScrollBar_offset.Name = "hScrollBar_offset";
			hScrollBar_offset.Size = new Size(614, 15);
			hScrollBar_offset.TabIndex = 5;
			// 
			// comboBox_cudaDevice
			// 
			comboBox_cudaDevice.FormattingEnabled = true;
			comboBox_cudaDevice.Location = new Point(12, 12);
			comboBox_cudaDevice.Name = "comboBox_cudaDevice";
			comboBox_cudaDevice.Size = new Size(240, 23);
			comboBox_cudaDevice.TabIndex = 6;
			comboBox_cudaDevice.SelectedIndexChanged += comboBox_cudaDevice_SelectedIndexChanged;
			// 
			// label_vram
			// 
			label_vram.AutoSize = true;
			label_vram.Font = new Font("Segoe UI", 9.75F, FontStyle.Regular, GraphicsUnit.Point,  0);
			label_vram.Location = new Point(12, 38);
			label_vram.Name = "label_vram";
			label_vram.Size = new Size(101, 17);
			label_vram.TabIndex = 7;
			label_vram.Text = "VRAM: 0 / 0 MB";
			// 
			// progressBar_vram
			// 
			progressBar_vram.Location = new Point(12, 58);
			progressBar_vram.Name = "progressBar_vram";
			progressBar_vram.Size = new Size(240, 10);
			progressBar_vram.TabIndex = 8;
			// 
			// listBox_kernels
			// 
			listBox_kernels.FormattingEnabled = true;
			listBox_kernels.ItemHeight = 15;
			listBox_kernels.Location = new Point(632, 12);
			listBox_kernels.Name = "listBox_kernels";
			listBox_kernels.Size = new Size(140, 184);
			listBox_kernels.TabIndex = 9;
			// 
			// label_kernelLoaded
			// 
			label_kernelLoaded.AutoSize = true;
			label_kernelLoaded.Font = new Font("Segoe UI", 8.25F, FontStyle.Regular, GraphicsUnit.Point,  0);
			label_kernelLoaded.Location = new Point(632, 199);
			label_kernelLoaded.Name = "label_kernelLoaded";
			label_kernelLoaded.Size = new Size(96, 13);
			label_kernelLoaded.TabIndex = 10;
			label_kernelLoaded.Text = "No kernel loaded";
			// 
			// groupBox_cudaMemory
			// 
			groupBox_cudaMemory.Controls.Add(label_chunkSize);
			groupBox_cudaMemory.Controls.Add(numericUpDown_chunkSize);
			groupBox_cudaMemory.Controls.Add(button_cudaMove);
			groupBox_cudaMemory.Location = new Point(632, 573);
			groupBox_cudaMemory.Name = "groupBox_cudaMemory";
			groupBox_cudaMemory.Size = new Size(140, 100);
			groupBox_cudaMemory.TabIndex = 11;
			groupBox_cudaMemory.TabStop = false;
			groupBox_cudaMemory.Text = "CUDA memory";
			// 
			// label_chunkSize
			// 
			label_chunkSize.AutoSize = true;
			label_chunkSize.Location = new Point(6, 53);
			label_chunkSize.Name = "label_chunkSize";
			label_chunkSize.Size = new Size(47, 15);
			label_chunkSize.TabIndex = 12;
			label_chunkSize.Text = "Chunks";
			// 
			// numericUpDown_chunkSize
			// 
			numericUpDown_chunkSize.Location = new Point(6, 71);
			numericUpDown_chunkSize.Maximum = new decimal(new int[] { 67108864, 0, 0, 0 });
			numericUpDown_chunkSize.Minimum = new decimal(new int[] { 1024, 0, 0, 0 });
			numericUpDown_chunkSize.Name = "numericUpDown_chunkSize";
			numericUpDown_chunkSize.Size = new Size(128, 23);
			numericUpDown_chunkSize.TabIndex = 12;
			numericUpDown_chunkSize.Value = new decimal(new int[] { 65536, 0, 0, 0 });
			numericUpDown_chunkSize.ValueChanged += numericUpDown_chunkSize_ValueChanged;
			// 
			// button_cudaMove
			// 
			button_cudaMove.Location = new Point(59, 42);
			button_cudaMove.Name = "button_cudaMove";
			button_cudaMove.Size = new Size(75, 23);
			button_cudaMove.TabIndex = 12;
			button_cudaMove.Text = "Move";
			button_cudaMove.UseVisualStyleBackColor = true;
			button_cudaMove.Click += button_cudaMove_Click;
			// 
			// groupBox_cudaFft
			// 
			groupBox_cudaFft.Controls.Add(button_cudaTransform);
			groupBox_cudaFft.Location = new Point(632, 467);
			groupBox_cudaFft.Name = "groupBox_cudaFft";
			groupBox_cudaFft.Size = new Size(140, 100);
			groupBox_cudaFft.TabIndex = 12;
			groupBox_cudaFft.TabStop = false;
			groupBox_cudaFft.Text = "CUDA FFT";
			// 
			// button_cudaTransform
			// 
			button_cudaTransform.Location = new Point(59, 71);
			button_cudaTransform.Name = "button_cudaTransform";
			button_cudaTransform.Size = new Size(75, 23);
			button_cudaTransform.TabIndex = 13;
			button_cudaTransform.Text = "Transform";
			button_cudaTransform.UseVisualStyleBackColor = true;
			button_cudaTransform.Click += button_cudaTransform_Click;
			// 
			// groupBox_cudaKernel
			// 
			groupBox_cudaKernel.Controls.Add(button_kernelRun);
			groupBox_cudaKernel.Controls.Add(label_kernelParam2);
			groupBox_cudaKernel.Controls.Add(numericUpDown_param2);
			groupBox_cudaKernel.Controls.Add(label_kernelParam1);
			groupBox_cudaKernel.Controls.Add(numericUpDown_param1);
			groupBox_cudaKernel.Controls.Add(button_kernelLoad);
			groupBox_cudaKernel.Controls.Add(button_kernelCompile);
			groupBox_cudaKernel.Location = new Point(632, 215);
			groupBox_cudaKernel.Name = "groupBox_cudaKernel";
			groupBox_cudaKernel.Size = new Size(140, 246);
			groupBox_cudaKernel.TabIndex = 13;
			groupBox_cudaKernel.TabStop = false;
			groupBox_cudaKernel.Text = "CUDA Kernel";
			// 
			// button_kernelRun
			// 
			button_kernelRun.Location = new Point(74, 217);
			button_kernelRun.Name = "button_kernelRun";
			button_kernelRun.Size = new Size(60, 23);
			button_kernelRun.TabIndex = 18;
			button_kernelRun.Text = "Run";
			button_kernelRun.UseVisualStyleBackColor = true;
			button_kernelRun.Click += button_kernelRun_Click;
			// 
			// label_kernelParam2
			// 
			label_kernelParam2.AutoSize = true;
			label_kernelParam2.Location = new Point(6, 115);
			label_kernelParam2.Name = "label_kernelParam2";
			label_kernelParam2.Size = new Size(57, 15);
			label_kernelParam2.TabIndex = 16;
			label_kernelParam2.Text = "Param #2";
			// 
			// numericUpDown_param2
			// 
			numericUpDown_param2.DecimalPlaces = 10;
			numericUpDown_param2.Enabled = false;
			numericUpDown_param2.Increment = new decimal(new int[] { 1, 0, 0, 196608 });
			numericUpDown_param2.Location = new Point(6, 133);
			numericUpDown_param2.Maximum = new decimal(new int[] { 999999, 0, 0, 327680 });
			numericUpDown_param2.Minimum = new decimal(new int[] { 1, 0, 0, 196608 });
			numericUpDown_param2.Name = "numericUpDown_param2";
			numericUpDown_param2.Size = new Size(128, 23);
			numericUpDown_param2.TabIndex = 17;
			numericUpDown_param2.Value = new decimal(new int[] { 1, 0, 0, 0 });
			// 
			// label_kernelParam1
			// 
			label_kernelParam1.AutoSize = true;
			label_kernelParam1.Location = new Point(6, 71);
			label_kernelParam1.Name = "label_kernelParam1";
			label_kernelParam1.Size = new Size(57, 15);
			label_kernelParam1.TabIndex = 14;
			label_kernelParam1.Text = "Param #1";
			// 
			// numericUpDown_param1
			// 
			numericUpDown_param1.DecimalPlaces = 10;
			numericUpDown_param1.Increment = new decimal(new int[] { 1, 0, 0, 196608 });
			numericUpDown_param1.Location = new Point(6, 89);
			numericUpDown_param1.Maximum = new decimal(new int[] { 999999, 0, 0, 327680 });
			numericUpDown_param1.Minimum = new decimal(new int[] { 1, 0, 0, 196608 });
			numericUpDown_param1.Name = "numericUpDown_param1";
			numericUpDown_param1.Size = new Size(128, 23);
			numericUpDown_param1.TabIndex = 14;
			numericUpDown_param1.Value = new decimal(new int[] { 1, 0, 0, 0 });
			// 
			// button_kernelLoad
			// 
			button_kernelLoad.Location = new Point(74, 22);
			button_kernelLoad.Name = "button_kernelLoad";
			button_kernelLoad.Size = new Size(60, 23);
			button_kernelLoad.TabIndex = 15;
			button_kernelLoad.Text = "Load";
			button_kernelLoad.UseVisualStyleBackColor = true;
			button_kernelLoad.Click += button_kernelLoad_Click;
			// 
			// button_kernelCompile
			// 
			button_kernelCompile.Location = new Point(6, 22);
			button_kernelCompile.Name = "button_kernelCompile";
			button_kernelCompile.Size = new Size(60, 23);
			button_kernelCompile.TabIndex = 14;
			button_kernelCompile.Text = "Compile";
			button_kernelCompile.UseVisualStyleBackColor = true;
			button_kernelCompile.Click += button_kernelCompile_Click;
			// 
			// textBox_kernelString
			// 
			textBox_kernelString.AcceptsReturn = true;
			textBox_kernelString.AcceptsTab = true;
			textBox_kernelString.Font = new Font("Bahnschrift SemiCondensed", 8.25F, FontStyle.Regular, GraphicsUnit.Point,  0);
			textBox_kernelString.Location = new Point(12, 215);
			textBox_kernelString.MaxLength = 2000000;
			textBox_kernelString.Multiline = true;
			textBox_kernelString.Name = "textBox_kernelString";
			textBox_kernelString.PlaceholderText = "Put your CUDA kernel code here and press compile.";
			textBox_kernelString.Size = new Size(614, 352);
			textBox_kernelString.TabIndex = 14;
			textBox_kernelString.WordWrap = false;
			// 
			// WindowMain
			// 
			AutoScaleDimensions = new SizeF(7F, 15F);
			AutoScaleMode = AutoScaleMode.Font;
			ClientSize = new Size(784, 921);
			Controls.Add(textBox_kernelString);
			Controls.Add(groupBox_cudaKernel);
			Controls.Add(groupBox_cudaFft);
			Controls.Add(groupBox_cudaMemory);
			Controls.Add(label_kernelLoaded);
			Controls.Add(listBox_kernels);
			Controls.Add(progressBar_vram);
			Controls.Add(label_vram);
			Controls.Add(comboBox_cudaDevice);
			Controls.Add(hScrollBar_offset);
			Controls.Add(label_trackMeta);
			Controls.Add(groupBox_controls);
			Controls.Add(pictureBox_waveform);
			Controls.Add(listBox_tracks);
			Controls.Add(listBox_log);
			MaximizeBox = false;
			MaximumSize = new Size(800, 960);
			MinimumSize = new Size(800, 960);
			Name = "WindowMain";
			Text = "CUDA Kernel with Precompiler for Audio-Processing";
			((System.ComponentModel.ISupportInitialize) pictureBox_waveform).EndInit();
			groupBox_controls.ResumeLayout(false);
			groupBox_controls.PerformLayout();
			groupBox_cudaMemory.ResumeLayout(false);
			groupBox_cudaMemory.PerformLayout();
			((System.ComponentModel.ISupportInitialize) numericUpDown_chunkSize).EndInit();
			groupBox_cudaFft.ResumeLayout(false);
			groupBox_cudaKernel.ResumeLayout(false);
			groupBox_cudaKernel.PerformLayout();
			((System.ComponentModel.ISupportInitialize) numericUpDown_param2).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_param1).EndInit();
			ResumeLayout(false);
			PerformLayout();
		}

		#endregion

		private ListBox listBox_log;
		private ListBox listBox_tracks;
		private PictureBox pictureBox_waveform;
		private GroupBox groupBox_controls;
		private TextBox textBox_timestamp;
		private Button button_playback;
		private Button button_colorGraph;
		private Label label_trackMeta;
		private Button button_colorBackground;
		private HScrollBar hScrollBar_offset;
		private ComboBox comboBox_cudaDevice;
		private Label label_vram;
		private ProgressBar progressBar_vram;
		private ListBox listBox_kernels;
		private Label label_kernelLoaded;
		private GroupBox groupBox_cudaMemory;
		private Label label_chunkSize;
		private NumericUpDown numericUpDown_chunkSize;
		private Button button_cudaMove;
		private GroupBox groupBox_cudaFft;
		private Button button_cudaTransform;
		private Button button_normalize;
		private GroupBox groupBox_cudaKernel;
		private Button button_kernelLoad;
		private Button button_kernelCompile;
		private Label label_kernelParam1;
		private NumericUpDown numericUpDown_param1;
		private TextBox textBox_kernelString;
		private Label label_kernelParam2;
		private NumericUpDown numericUpDown_param2;
		private Button button_kernelRun;
	}
}
