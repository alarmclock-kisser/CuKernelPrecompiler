using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using ManagedCuda.VectorTypes;
using System.Diagnostics;
using System.Drawing;

namespace CuKernelPrecompiler
{
	public class CudaKernelPrecompiler
	{
		private string Repopath;
		private ListBox Logbox;
		private ListBox KernelsList;
		private Label KernelLabel;
		private Button CompileButton;
		private Button LoadButton;
		private Button RunButton;
		private NumericUpDown Param1;
		private NumericUpDown Param2;
		private TextBox KernelString;

		public PrimaryContext Ctx;
		public List<KernelObject> Kernels = [];

		public CudaKernel? Kernel => Kernels.FirstOrDefault(k => k.Name == KernelLabel.Text)?.Kernel;


		public CudaKernelPrecompiler(string repopath, PrimaryContext ctx, ListBox listBox_log, ListBox listBox_kernels, Label label_kernelLoaded, Button button_kernelCompile, Button button_kernelLoad, Button button_kernelRun, NumericUpDown numericUpDown_param1, NumericUpDown numericUpDown_param2, TextBox textBox_kernelString)
		{
			this.Repopath = repopath;
			this.Ctx = ctx;
			this.Logbox = listBox_log;
			this.KernelsList = listBox_kernels;
			this.KernelLabel = label_kernelLoaded;
			this.CompileButton = button_kernelCompile;
			this.LoadButton = button_kernelLoad;
			this.RunButton = button_kernelRun;
			this.Param1 = numericUpDown_param1;
			this.Param2 = numericUpDown_param2;
			this.KernelString = textBox_kernelString;

			// Register events
			CompileButton.Click += (sender, e) => PrecompileKernelString();
			KernelString.TextChanged += (sender, e) => AdjustKernelStringBox();

		}



		public void Log(string message, string inner = "", int layer = 1, bool update = false)
		{
			string msg = "[" + DateTime.Now.ToString("HH:mm:ss.fff") + "] ";
			msg += "<Kernel>";

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
				Logbox.Items[Logbox.Items.Count - 1] = msg;
			}
			else
			{
				Logbox.Items.Add(msg);
				Logbox.SelectedIndex = Logbox.Items.Count - 1;
			}
		}

		public void AdjustKernelStringBox()
		{
			// If text has more than 15 lines, enable vertical scroll bar
			KernelString.ScrollBars = KernelString.Lines.Length > 25 ? ScrollBars.Vertical : ScrollBars.None;
		}

		public void PrecompileKernelString()
		{
			// Get kernel string
			string kernelString = KernelString.Text;

			// Abort if too short
			if (kernelString.Length < 10)
			{
				Log("Kernel string too short", "Abort", 1);
				return;
			}

			// Get kernel name (string between __global__ void and (...)
			string kernelName = kernelString.Split(["__global__ void ", "("], StringSplitOptions.None)[1];

			// Get kernel parameters (string between ( and ))
			string[] kernelParams = kernelString.Split(["(", ")"], StringSplitOptions.None)[1].Split([","], StringSplitOptions.None);
			for (int i = 0; i < kernelParams.Length; i++)
			{
				kernelParams[i] = kernelParams[i].Trim().Split([" "], StringSplitOptions.None).First();
				Log("Param " + i + ": " + kernelParams[i], "", 2);
			}

			// Compile kernel
			string ptxPath = CompileString(kernelString);

			// Abort if compilation failed
			if (ptxPath == "")
			{
				Log("Kernel compilation failed", "Abort", 1);
				return;
			}

			// Load kernel
			CudaKernel? kernel = LoadKernel(ptxPath, true);

			// Abort if loading failed
			if (kernel == null)
			{
				Log("Kernel loading failed", "Abort", 1);
				return;
			}

			// Add kernel to list
			Kernels.Add(new KernelObject(kernelName, ptxPath, kernel));

			Param1.Enabled = false;
			Param2.Enabled = false;
			// Toggle param1&2 enabled (first 2 are always int and array)
			if (kernelParams.Length > 2)
			{
				Param1.Enabled = true;
				if (kernelParams[2] == "float")
				{
					Param1.DecimalPlaces = 10;
				}
				else if (kernelParams[2] == "double")
				{
					Param1.DecimalPlaces = 20;
				}
				else if (kernelParams[2] == "int")
				{
					Param1.DecimalPlaces = 0;
				}
				else
				{
					Param1.DecimalPlaces = 0;
				}
			}
			if (kernelParams.Length > 3)
			{
				Param2.Enabled = true;
				if (kernelParams[3] == "float")
				{
					Param2.DecimalPlaces = 10;
				}
				else if (kernelParams[3] == "double")
				{
					Param2.DecimalPlaces = 20;
				}
				else if (kernelParams[3] == "int")
				{
					Param2.DecimalPlaces = 0;
				}
				else
				{
					Param2.DecimalPlaces = 0;
				}
			}

			// Set params in KernelObject
			Kernels.FirstOrDefault(k => k.Name == kernelName).Params = kernelParams;
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

		public CudaKernel? LoadKernel(string filepath, bool silent = false)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return null;
			}

			// Var for kernel
			CudaKernel? kernel = null;

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
				kernel = Ctx.LoadKernelPTX(ptxCode, kernelName);
			}
			catch (Exception ex)
			{
				if (!silent)
				{
					Log("Failed to load kernel " + kernelName, ex.Message, 1);
					string logMsg = ex.Message + Environment.NewLine + Environment.NewLine + ex.InnerException?.Message ?? "";
					File.WriteAllText(logpath, logMsg);
				}
				kernel = null;
			}
			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			if (!silent)
			{
				Log("Kernel loaded within " + deltaMicros.ToString("N0") + " µs", "", 2);
			}

			KernelLabel.Text = kernel?.KernelName ?? "None loaded";

			return kernel;
		}


		public void RunKernel<T>(CudaDeviceVariable<T> variable) where T : unmanaged
		{
			// Get kernel
			CudaKernel? kernel = Kernel;

			// Abort if no kernel loaded
			if (kernel == null)
			{
				Log("No kernel loaded", "Abort", 1);
				return;
			}

			// Get kernel name
			string kernelName = kernel.KernelName;

			// Get kernel parameters definition (Name and Types from Kernels list)
			string[] kernelParamsDefinition = Kernels.FirstOrDefault(k => k.Name == kernelName)?.Params ?? [];

			// Create a list to hold the dynamic kernel parameters
			List<object> dynamicParams =
			[
				// Add size parameter (always needed as first parameter)
				variable.Size,
				// Add variable.DevicePointer (always needed as second parameter)
				variable.DevicePointer,
			];

			// Parameter 1 Handling
			if (Param1.Enabled && kernelParamsDefinition.Length > 2)
			{
				dynamicParams.Add(ConvertParameter(kernelParamsDefinition[2], Param1.Value));
			}

			// Parameter 2 Handling
			if (Param2.Enabled && kernelParamsDefinition.Length > 3)
			{
				dynamicParams.Add(ConvertParameter(kernelParamsDefinition[3], Param2.Value));
			}

			// Check cuda variable type
			if (variable.Size < 1)
			{
				Log("CUDA variable empty.", "Abort", 1);
				return;
			}

			// Get type of T as string
			string varType = typeof(T) == typeof(float) ? "float" :
							  typeof(T) == typeof(float2) ? "float2" : "unsupported";

			if (varType == "unsupported" || kernelParamsDefinition.Length < 2 || kernelParamsDefinition[1] != varType)
			{
				Log("CUDA variable type not supported.", "Abort", 1);
				return;
			}

			// Run kernel with dynamic parameter list
			Stopwatch sw = Stopwatch.StartNew();
			kernel.Run(dynamicParams.ToArray());
			sw.Stop();

			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			Log($"Kernel '{kernelName}' executed on variable of type '{varType}' and size {variable.Size} within {deltaMicros:N0} µs", "", 2, true);
		}

		private object ConvertParameter(string type, object value)
		{
			return type switch
			{
				"int" => Convert.ToInt32(value),
				"float" => Convert.ToSingle(value),
				"double" => Convert.ToDouble(value),
				_ => throw new ArgumentException($"Unsupported parameter type: {type}")
			};
		}











	}






	public class KernelObject
	{
		public string Name;
		public string Path;
		public CudaKernel? Kernel;
		public string[] Params = [];

		public KernelObject(string name, string path, CudaKernel? kernel)
		{
			this.Name = name;
			this.Path = path;
			this.Kernel = kernel;
		}
	}
}
