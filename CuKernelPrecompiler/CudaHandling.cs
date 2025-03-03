using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using ManagedCuda.NVRTC;
using ManagedCuda.VectorTypes;
using System.Diagnostics;

namespace CuKernelPrecompiler
{
	public class CudaHandling
	{
		// ~~~~~ ~~~~~ ~~~~~ ATTRIBUTES ~~~~~ ~~~~~ ~~~~~ \\
		public string Repopath;
		public ListBox LogBox;
		public ComboBox DevicesBox;
		public Label VramLabel;
		public ProgressBar VramBar;
		public ListBox KernelBox;
		public Label KernelLabel;

		public int DeviceId => DevicesBox.SelectedIndex;

		public PrimaryContext? Ctx = null;

		public CudaKernel? Kernel = null;

		public Dictionary<long, List<CudaDeviceVariable<float>>> DeviceFloatVars = [];
		public Dictionary<long, List<CudaDeviceVariable<float2>>> DeviceComplexVars = [];


		public int LogStep = 10000;

		// ~~~~~ ~~~~~ ~~~~~ CONSTRUCTOR ~~~~~ ~~~~~ ~~~~~ \\
		public CudaHandling(string repopath, ListBox? logbox = null, ComboBox? devicesBox = null, Label? vramLabel = null, ProgressBar? vramBar = null, ListBox? kernelBox = null, Label? kernelLabel = null)
		{
			Repopath = repopath;
			LogBox = logbox ?? new ListBox();
			DevicesBox = devicesBox ?? new ComboBox();
			VramLabel = vramLabel ?? new Label();
			VramBar = vramBar ?? new ProgressBar();
			KernelBox = kernelBox ?? new ListBox();
			KernelLabel = kernelLabel ?? new Label();

			FillDevices();
			DevicesBox.SelectedIndexChanged += (sender, e) => InitContext();

			// Set default device
			DevicesBox.SelectedIndex = 0;

			// Verify compiled kernels
			VerifyCompiledKernels(KernelBox, true);
			UnloadKernel(true);

			// Register events


		}







		// ~~~~~ ~~~~~ ~~~~~ METHODS ~~~~~ ~~~~~ ~~~~~ \\
		public void Log(string message, string inner = "", int layer = 1, bool update = false)
		{
			string msg = "[" + DateTime.Now.ToString("HH:mm:ss.fff") + "] ";
			msg += "<CUDA>";

			for (int i = 0; i <= layer; i++)
			{
				msg += " - ";
			}

			msg += message;

			if (inner != "")
			{
				msg += "  (" + inner + ")";
			}

			if (update)
			{
				LogBox.Items[LogBox.Items.Count - 1] = msg;
			}
			else
			{
				LogBox.Items.Add(msg);
				LogBox.SelectedIndex = LogBox.Items.Count - 1;
			}
		}

		public string[] FillDevices()
		{
			string[] names = new string[CudaContext.GetDeviceCount()];

			try
			{
				for (int i = 0; i < names.Length; i++)
				{
					names[i] = CudaContext.GetDeviceName(i);
				}
			}
			catch
			(Exception e)
			{
				Log(e.Message, e.InnerException?.Message ?? "", 1);
			}

			// Update UI
			DevicesBox.Items.Clear();
			DevicesBox.Items.AddRange(names);
			DevicesBox.Items.Add(" - Use no CUDA device - ");

			return names;
		}

		public int[] GetVramInfo()
		{
			// Abort if no device selected
			if (Ctx == null)
			{
				VramLabel.Text = "No device selected";
				VramBar.Value = 0;
				return [0, 0, 0];
			}

			// Get device memory info
			long[] usage = [0, 0, 0];
			try
			{
				usage[0] = Ctx.GetTotalDeviceMemorySize() / 1024 / 1024;
				usage[1] = Ctx.GetFreeDeviceMemorySize() / 1024 / 1024;
				usage[2] = usage[0] - usage[1];
			}
			catch (Exception e)
			{
				Log("Failed to get VRAM info", e.Message, 1);
			}

			// Update UI
			VramLabel.Text = $"VRAM: {usage[2]} MB / {usage[0]} MB";
			VramBar.Maximum = (int) usage[0];
			VramBar.Value = (int) usage[2];

			// Return info
			return [(int) usage[0], (int) usage[1], (int) usage[2]];
		}

		public void InitContext()
		{
			// Dispose previous context
			Dispose();

			// Abort if no device selected
			if (DeviceId == -1 || DeviceId >= CudaContext.GetDeviceCount())
			{
				Log("No device selected", "", 1);
				GetVramInfo();
				return;
			}

			// Create context
			try
			{
				Ctx = new PrimaryContext(DeviceId);
				Ctx.SetCurrent();
				Log("Context created", Ctx.GetDeviceName(), 1);

				// Update VRAM info
				GetVramInfo();
			}
			catch (Exception e)
			{
				Log("Failed to create context", e.Message, 1);
			}
		}

		public void Dispose()
		{
			Ctx?.Dispose();
			Ctx = null;
		}

		public long PushChunks<T>(List<T[]> chunks) where T : unmanaged
		{
			long firstPointer = 0;
			List<CudaDeviceVariable<T>> devChunks = new();

			Stopwatch stopwatch = Stopwatch.StartNew();
			Log("Started pushing " + chunks.Count + " chunks of type " + typeof(T).Name);

			int c = LogStep - 1;
			foreach (var chunk in chunks)
			{
				int length = chunk.Length;

				// Allocate memory
				CudaDeviceVariable<T> devChunk = new(length);
				devChunks.Add(devChunk);

				// Copy data
				devChunk.CopyToDevice(chunk);

				
				c++;
				if (c %  LogStep == 0)
				{
					Log("Pushed chunk " + devChunks.Count + " / " + chunks.Count, "", 2, true);
				}
			}

			// Get first pointer
			if (devChunks.Count > 0)
			{
				firstPointer = devChunks[0].DevicePointer.Pointer;
			}

			// Add to corresponding dictionary
			if (typeof(T) == typeof(float))
			{
				DeviceFloatVars[firstPointer] = devChunks.Cast<CudaDeviceVariable<float>>().ToList();
			}
			else if (typeof(T) == typeof(float2))
			{
				DeviceComplexVars[firstPointer] = devChunks.Cast<CudaDeviceVariable<float2>>().ToList();
			}

			stopwatch.Stop();
			long deltaMicro = stopwatch.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));


			Log("Pushed " + chunks.Count + " chunks within " + deltaMicro.ToString("N0") + " µs.", "", 1, true);

			GetVramInfo();
			return firstPointer;
		}

		public List<T[]> PullChunks<T>(long firstPointer) where T : unmanaged
		{
			List<T[]> chunks = new();

			Stopwatch stopwatch = Stopwatch.StartNew();
			Log("Started pulling " + chunks.Count + " chunks of type " + typeof(T).Name);

			if (typeof(T) == typeof(float))
			{
				if (DeviceFloatVars.ContainsKey(firstPointer))
				{
					int c = LogStep - 1;
					foreach (var devChunk in DeviceFloatVars[firstPointer])
					{
						int length = devChunk.Size;
						T[] chunk = new T[length];

						// Copy data
						devChunk.CopyToHost(chunk);
						chunks.Add(chunk);

						// Dispose device memory
						devChunk.Dispose();

						c++;
						if (c %  LogStep == 0)
						{
							Log("Pulled chunk " + chunks.Count + " / " + DeviceFloatVars[firstPointer].Count, "", 2, true);
						}
					}

					// Remove from dictionary
					DeviceFloatVars.Remove(firstPointer);
				}
			}
			else if (typeof(T) == typeof(float2))
			{
				if (DeviceComplexVars.ContainsKey(firstPointer))
				{
					int c = LogStep - 1;
					foreach (var devChunk in DeviceComplexVars[firstPointer])
					{
						int length = devChunk.Size;
						T[] chunk = new T[length];

						// Copy data
						devChunk.CopyToHost(chunk);
						chunks.Add(chunk);

						// Dispose device memory
						devChunk.Dispose();

						c++;
						if (c %  LogStep == 0)
						{
							Log("Pulled chunk " + chunks.Count + " / " + DeviceComplexVars[firstPointer].Count, "", 2, true);
						}
					}

					// Remove from dictionary
					DeviceComplexVars.Remove(firstPointer);
				}
			}

			stopwatch.Stop();
			long deltaMicros = stopwatch.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			Log("Pulled " + chunks.Count + " chunks within " + deltaMicros.ToString("N0") + " µs.", "", 1, true);

			GetVramInfo();
			return chunks;
		}

		public List<CudaDeviceVariable<float>> GetFloatVars(long firstPointer, bool silent = false)
		{
			if (DeviceFloatVars.ContainsKey(firstPointer))
			{
				return DeviceFloatVars[firstPointer];
			}
			if (DeviceFloatVars.ContainsKey(-1 * firstPointer))
			{
				return DeviceFloatVars[-1 * firstPointer];
			}

			if (!silent)
			{
				Log("No variables found", "<" + Math.Abs(firstPointer) + ">", 1);
			}
			return [];
		}

		public List<CudaDeviceVariable<float2>> GetComplexVars(long firstPointer, bool silent = false)
		{
			if (DeviceComplexVars.ContainsKey(firstPointer))
			{
				return DeviceComplexVars[firstPointer];
			}
			if (DeviceComplexVars.ContainsKey(-1 * firstPointer))
			{
				return DeviceComplexVars[-1 * firstPointer];
			}

			if (!silent)
			{
				Log("No variables found", "<" + Math.Abs(firstPointer) + ">", 1);
			}
			return [];
		}

		public long ClearMemory(long firstPointer, bool silent = false)
		{
			firstPointer = Math.Abs(firstPointer);
			long freed = 0;

			if (DeviceFloatVars.ContainsKey(firstPointer))
			{
				foreach (var devChunk in DeviceFloatVars[firstPointer])
				{
					freed += devChunk.Size;
					devChunk.Dispose();
				}
				DeviceFloatVars.Remove(firstPointer);
			}
			if (DeviceComplexVars.ContainsKey(firstPointer))
			{
				foreach (var devChunk in DeviceComplexVars[firstPointer])
				{
					freed += devChunk.Size;
					devChunk.Dispose();
				}
				DeviceComplexVars.Remove(firstPointer);
			}

			GC.Collect();

			freed *= sizeof(float);
			if (!silent)
			{
				Log("Cleared pointer-group <" + firstPointer + ">", (freed / 1024 / 1024) + " MB", 1);
			}

			return freed;
		}

		public void ClearAllMemory(bool silent = false)
		{
			long totalFreed = 0;

			foreach (var devChunks in DeviceFloatVars.Values)
			{
				foreach (var devChunk in devChunks)
				{
					totalFreed += devChunk.Size;
					long pointer = devChunk.DevicePointer.Pointer;
					devChunk.Dispose();

					if (!silent)
					{
						Log("Found & cleared float pointer ", "<" + pointer + ">", 2);
					}
				}
			}
			DeviceFloatVars.Clear();

			foreach (var devChunks in DeviceComplexVars.Values)
			{
				foreach (var devChunk in devChunks)
				{
					totalFreed += devChunk.Size;
					long pointer = devChunk.DevicePointer.Pointer;
					devChunk.Dispose();

					if (!silent)
					{
						Log("Found & cleared complex pointer ", "<" + pointer + ">", 2);
					}
				}
			}
			DeviceComplexVars.Clear();

			GC.Collect();
			totalFreed *= sizeof(float);

			if (!silent)
			{
				Log("Cleared all memory", (totalFreed / 1024 / 1024) + " MB", 1);
			}
		}

		public long PerformFFT(long firstPointer)
		{
			if (firstPointer == 0)
			{
				Log("No input variables found", "", 1);
				return 0;
			}

			List<CudaDeviceVariable<float>> inputVars = GetFloatVars(firstPointer);
			List<CudaDeviceVariable<float2>> outputVars = [];

			Stopwatch sw = Stopwatch.StartNew();
			Log("Performing FFT on " + inputVars.Count + " input variables");

			int c = LogStep - 1;
			// Perform FFT on each input
			for (int i = 0; i < inputVars.Count; i++)
			{
				// Create output variable
				CudaDeviceVariable<float2> outputVar = new CudaDeviceVariable<float2>(inputVars[i].Size);

				// Create plan
				var plan = new CudaFFTPlan1D(inputVars[i].Size, cufftType.R2C, 1);

				// Execute FFT
				plan.Exec(inputVars[i].DevicePointer, outputVar.DevicePointer);

				// Destroy plan
				plan.Dispose();

				// Add to output list
				outputVars.Add(outputVar);

				// Log progress
				c++;
				if (c %  LogStep == 0)
				{
					Log("Performed FFT on input " + i + " / " + inputVars.Count, "", 2, true);
				}
			}


			sw.Stop();
			long deltaMicros = (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L)));
			Log("Performed FFT on " + inputVars.Count + " input variables within " + deltaMicros.ToString("N0") + " µs", "", 1, true);

			ClearMemory(firstPointer, true);
			GC.Collect();
			GetVramInfo();

			firstPointer = outputVars[0].DevicePointer.Pointer;
			DeviceComplexVars[firstPointer] = outputVars;
			return firstPointer;
		}

		public long PerformIFFT(long firstPointer)
		{
			if (firstPointer == 0)
			{
				Log("No input variables found", "", 1);
				return 0;
			}

			List<CudaDeviceVariable<float2>> inputVars = GetComplexVars(firstPointer);
			List<CudaDeviceVariable<float>> outputVars = [];

			Stopwatch sw = Stopwatch.StartNew();
			Log("Performing IFFT on " + inputVars.Count + " input variables");

			int c = LogStep - 1;
			// Perform IFFT on each input
			for (int i = 0; i < inputVars.Count; i++)
			{
				// Create output variable
				CudaDeviceVariable<float> outputVar = new CudaDeviceVariable<float>(inputVars[i].Size);

				// Create plan
				var plan = new CudaFFTPlan1D(inputVars[i].Size, cufftType.C2R, 1);

				// Execute IFFT
				plan.Exec(inputVars[i].DevicePointer, outputVar.DevicePointer);

				// Destroy plan
				plan.Dispose();

				// Add to output list
				outputVars.Add(outputVar);

				// Log progress
				c++;
				if (c %  LogStep == 0)
				{
					Log("Performed IFFT on input " + i + " / " + inputVars.Count, "", 2, true);
				}
			}

			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			Log("Performed IFFT on " + inputVars.Count + " input variables within " + deltaMicros.ToString("N0") + " µs", "", 1, true);

			ClearMemory(firstPointer, true);
			GC.Collect();
			GetVramInfo();

			firstPointer = outputVars[0].DevicePointer.Pointer;
			DeviceFloatVars[firstPointer] = outputVars;
			return firstPointer;
		}

		public string CompileKernel(string filepath)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return "";
			}

			string kernelName = Path.GetFileNameWithoutExtension(filepath);
			kernelName = kernelName.Replace("Kernel", "");

			string logpath = Path.Combine(Repopath, "Resources\\Logs", kernelName + "Kernel" + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			Log("Compiling kernel " + kernelName);

			// Load kernel file
			string kernelCode = File.ReadAllText(filepath);


			var rtc = new CudaRuntimeCompiler(kernelCode, kernelName);

			try
			{
				// Compile kernel
				rtc.Compile([]);

				if (rtc.GetLogAsString().Length > 0)
				{
					Log("Kernel compiled with warnings", "", 1);
					File.WriteAllText(logpath, rtc.GetLogAsString());
				}


				sw.Stop();
				long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
				Log("Kernel compiled within " + deltaMicros.ToString("N0") + " µs", "Repo\\" + Path.GetRelativePath(Repopath, logpath), 2, true);

				// Get ptx code
				byte[] ptxCode = rtc.GetPTX();

				// Export ptx
				string ptxPath = Path.Combine(Repopath, "Resources\\Kernels\\PTX", kernelName + "Kernel" + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				Log("PTX code exported to " + ptxPath, "", 1);

				return ptxPath;
			}
			catch (Exception ex)
			{
				File.WriteAllText(logpath, rtc.GetLogAsString());
				Log(ex.Message, ex.InnerException?.Message ?? "", 1);

				return "";
			}

		}

		public string CompileString(string kernelString)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return "";
			}

			string kernelName = kernelString.Split("__global__ void ")[1].Split("(")[0];
			kernelName = kernelName.Replace("Kernel", "");

			string logpath = Path.Combine(Repopath, "Resources\\Logs", kernelName + "Kernel" + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			Log("Compiling kernel " + kernelName);

			// Load kernel file
			string kernelCode = kernelString;

			// Save also the kernel string as .c file
			string cPath = Path.Combine(Repopath, "Resources\\Kernels\\CU", kernelName + "Kernel" + ".c");
			File.WriteAllText(cPath, kernelCode);


			var rtc = new CudaRuntimeCompiler(kernelCode, kernelName);

			try
			{
				// Compile kernel
				rtc.Compile([]);

				if (rtc.GetLogAsString().Length > 0)
				{
					Log("Kernel compiled with warnings", "", 1);
					File.WriteAllText(logpath, rtc.GetLogAsString());
				}


				sw.Stop();
				long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
				Log("Kernel compiled in " + deltaMicros.ToString("N0") + " µs", "Repo\\" + Path.GetRelativePath(Repopath, logpath), 2, true);

				// Get ptx code
				byte[] ptxCode = rtc.GetPTX();

				// Export ptx
				string ptxPath = Path.Combine(Repopath, "Resources\\Kernels\\PTX", kernelName + "Kernel" + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				Log("PTX code exported to " + ptxPath, "", 1);

				return ptxPath;
			}
			catch (Exception ex)
			{
				File.WriteAllText(logpath, rtc.GetLogAsString());
				Log(ex.Message, ex.InnerException?.Message ?? "", 1);

				return "";
			}
		}

		public void CompileAll(string kernelDir)
		{
			// Get all kernel files
			string[] kernelFiles = Directory.GetFiles(kernelDir, "*.c");

			// Compile each kernel
			foreach (var file in kernelFiles)
			{
				CompileKernel(file);
			}

			// Verify compiled kernels	
			VerifyCompiledKernels(KernelBox, true);
			UnloadKernel(true);
		}

		public void LoadKernel(string filepath, bool silent = false)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return;
			}

			// Get kernel name
			string kernelName = Path.GetFileNameWithoutExtension(filepath);
			kernelName = kernelName.Replace("Kernel", "");

			// Get log path
			string logpath = Path.Combine(Repopath, "Resources\\Logs", kernelName + "Kernel" + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			if (!silent)
			{
				Log("Started loading kernel " + kernelName);
			}

			try
			{
				// Load ptx code
				byte[] ptxCode = File.ReadAllBytes(filepath);

				// Load kernel
				Kernel = Ctx.LoadKernelPTX(ptxCode, kernelName);
			}
			catch (Exception ex)
			{
				if (!silent)
				{
					Log("Failed to load kernel " + kernelName, ex.Message, 1);
					string logMsg = ex.Message + Environment.NewLine + Environment.NewLine + ex.InnerException?.Message ?? "";
					File.WriteAllText(logpath, logMsg);
				}
				Kernel = null;
			}
			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			if (!silent)
			{
				Log("Kernel loaded within " + deltaMicros.ToString("N0") + " µs", "", 2);
			}

			KernelLabel.Text = Kernel?.KernelName ?? "None loaded";
		}

		public void LoadKernelByName(string kernelName, bool silent = false)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return;
			}

			// Abort if no kernel name
			if (kernelName == "")
			{
				if (!silent)
				{
					Log("No kernel name provided", "", 1);
				}
				return;
			}

			// Get kernel path
			string kernelPath = Path.Combine(Repopath, "Resources\\Kernels\\PTX", kernelName + "Kernel.ptx");

			LoadKernel(kernelPath, silent);
		}

		public void UnloadKernel(bool silent = false)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return;
			}


			Kernel = null;

			if (!silent)
			{
				Log("Kernel unloaded", "", 1);
			}

			KernelLabel.Text = "None loaded";

		}

		public List<string> VerifyCompiledKernels(ListBox? kernelsListbox = null, bool silent = false)
		{
			// Get all kernel files
			List<string> validKernelPaths = [];
			string kernelsPath = Path.Combine(Repopath, "Resources\\Kernels\\PTX");
			List<FileInfo> kernelFiles = new DirectoryInfo(kernelsPath).GetFiles("*.ptx").ToList();

			// Verify listbox
			kernelsListbox = kernelsListbox ?? new ListBox();
			kernelsListbox.Items.Clear();

			// Try to load each kernel -> add name to listbox
			foreach (var file in kernelFiles)
			{
				try
				{
					LoadKernel(file.FullName, true);

					if (Kernel != null)
					{
						kernelsListbox.Items.Add(Kernel.KernelName);
					}

					validKernelPaths.Add(file.FullName);

					if (!silent)
					{
						Log("Loaded kernel '" + file.Name + "'", "", 2);
					}

					Kernel = null;
				}
				catch (Exception ex)
				{
					if (!silent)
					{
						Log("Failed to try-load kernel '" + file.Name + "'", ex.Message + " (" + ex.InnerException?.Message ?? "" + ")", 2);
					}
				}
			}

			kernelsListbox.Items.Clear();
			foreach (var path in validKernelPaths)
			{
				kernelsListbox.Items.Add(Path.GetFileNameWithoutExtension(path).Replace("Kernel", ""));
			}

			return validKernelPaths;
		}

		public List<string> GetCompiledKernelPaths()
		{
			// Get all kernel files
			List<string> validKernelPaths = [];
			string kernelsPath = Path.Combine(Repopath, "Resources\\Kernels\\PTX");
			List<FileInfo> kernelFiles = new DirectoryInfo(kernelsPath).GetFiles("*.ptx").ToList();

			// Add to list
			foreach (var file in kernelFiles)
			{
				validKernelPaths.Add(file.FullName);
			}
			return validKernelPaths;
		}

		public void LoadKernelFromList(List<string> validKernelPaths, int index)
		{
			if (index < 0 || index >= validKernelPaths.Count)
			{
				Log("Invalid index", "", 1);
				return;
			}

			LoadKernel(validKernelPaths[index], true);
		}

		private void DeleteKernelByEntry(string entry)
		{
			if (string.IsNullOrWhiteSpace(entry))
			{
				Log("No valid kernel selected", "", 1);
				return;
			}

			// Datei-Pfad generieren
			string kernelPath = Path.Combine(Repopath, "Resources", "Kernels", "PTX", entry + "Kernel.ptx");

			if (!File.Exists(kernelPath))
			{
				Log($"Kernel file '{entry}' not found", "", 1);
				return;
			}

			try
			{
				File.Delete(kernelPath);
				Log($"Kernel '{entry}' deleted successfully", "", 1);
			}
			catch (Exception ex)
			{
				Log($"Error deleting kernel '{entry}'", ex.Message, 1);
				return;
			}

			// GUI-Update nach erfolgreichem Löschen
			VerifyCompiledKernels(KernelBox, true);
			UnloadKernel(true);
		}

		public void RunKernel(long firstPointer, float param1, float? param2 = null)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return;
			}
			if (Kernel == null)
			{
				Log("No kernel loaded", "", 1);
				return;
			}
			if (firstPointer == 0)
			{
				Log("No input variables found", "", 1);
				return;
			}

			// Get input variables
			List<CudaDeviceVariable<float>> inputFloatVars = GetFloatVars(firstPointer, true);
			List<CudaDeviceVariable<float2>> inputComplexVars = GetComplexVars(firstPointer, true);

			Dictionary<CUdeviceptr, long> pointers = [];

			// Get pointers
			for (int i = 0; i < inputFloatVars.Count; i++)
			{
				pointers[inputFloatVars[i].DevicePointer] = inputFloatVars[i].Size;
			}
			for (int i = 0; i < inputComplexVars.Count; i++)
			{
				pointers[inputComplexVars[i].DevicePointer] = inputComplexVars[i].Size;
			}

			// Set grid and block size
			Kernel.BlockDimensions = new dim3(256, 1, 1);
			Kernel.GridDimensions = new dim3((int) Math.Ceiling(pointers.Count / 256.0), 1, 1);

			// Run kernel for each pointer
			Stopwatch sw = Stopwatch.StartNew();
			Log("Started running kernel " + Kernel.KernelName + " on " + pointers.Count + " input variables");

			int c = LogStep - 1;
			// Run for each pointer
			for (int i = 0; i < pointers.Count; i++)
			{
				if (param2 == null)
				{
					Kernel.Run(pointers.Values.ElementAt(i), pointers.Keys.ElementAt(i), param1);
				}
				else
				{
					Kernel.Run(pointers.Values.ElementAt(i), pointers.Keys.ElementAt(i), param1, param2.Value);
				}

				// Log progress
				c++;
				if (c %  LogStep == 0)
				{
					Log("Ran kernel on input " + i + " / " + pointers.Count, "", 2, true);
				}
			}

			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			Log("Ran kernel on " + pointers.Count + " input variables within " + deltaMicros.ToString("N0") + " µs", "", 1, true);

			// Vram info
			GetVramInfo();
		}

	}
}
