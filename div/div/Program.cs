using System;
using System.Collections.Generic;
using System.IO;

namespace div
{
    class FingerPrinter
    {
        public static string[] FPname = { "Descriptor", "FP", "ExtFP" };//, "MACCSFP", "PubchemFP", "SubFP" };
        public static int[] FPcount = { 729, 1024, 1024 };//, 166, 881, 307 };
        public string[][] AllFP = new string[3][];
        public string name;
        public FingerPrinter()
        {
            for (int i = 0; i < AllFP.GetLength(0); i++)
            {
                AllFP[i] = new string[FPcount[i]];
            }
        }
    }
    class Program
    {
        //思路错误
        static List<string> fromOutput()
        {
            string path = @"C:\Users\lenovo\Desktop\毕业论文\result\骨骼系统.txt";
            string fileformat = @"C:\Users\lenovo\Desktop\毕业论文\result\smiles\{0}&{1}";
            string sfile;
            string cas, srn;
            List<string> smiles = new List<string>();
            using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read))
            {
                StreamReader sr = new StreamReader(fs);
                while (!string.IsNullOrEmpty(srn = sr.ReadLine()))
                {
                    cas = sr.ReadLine();
                    sfile = string.Format(fileformat, cas, srn);
                    if (File.Exists(sfile))
                    {
                        using (FileStream sfs = new FileStream(sfile, FileMode.Open, FileAccess.Read))
                        {
                            StreamReader ssr = new StreamReader(sfs);
                            smiles.Add(ssr.ReadLine());
                        }
                    }
                }
            }
            return smiles;
        }
        static List<string> fromCSV()
        {
            string path = @"C:\Users\lenovo\Desktop\毕业论文\result\ames.csv";
            string line;
            string[] compond;
            List<string> smiles = new List<string>();

            using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read))
            {
                StreamReader sr = new StreamReader(fs);
                sr.ReadLine();
                while (!string.IsNullOrEmpty(line = sr.ReadLine()))
                {
                    compond = line.Split(',');
                    if (compond[1] == "0")
                    {
                        smiles.Add(compond[0]);
                    }
                }
            }
            return smiles;
        }
        //temp
        //toTab("tox0.fps-train0-SubFP", "nontox.fps-train0-SubFP", "train-SubFP.tab");
        //toTab("tox0.fps-test0-SubFP", "nontox.fps-test0-SubFP", "test-SubFP.tab");
        static void toTab(string toxfile, string nontoxfile, string outputname)
        {
            string root = @"C:\Users\lenovo\Desktop\毕业论文\result\";
            List<string[]> list = new List<string[]>();
            string line = "";
            using (FileStream fs = new FileStream(root + toxfile, FileMode.Open, FileAccess.Read))
            {
                StreamReader sr = new StreamReader(fs);
                while (!string.IsNullOrEmpty((line = sr.ReadLine())))
                {
                    string[] now = line.Trim().Split(',');
                    now[0] = "1";
                    list.Add(now);
                }
            }
            using (FileStream fs = new FileStream(root + nontoxfile, FileMode.Open, FileAccess.Read))
            {
                StreamReader sr = new StreamReader(fs);
                while (!string.IsNullOrEmpty((line = sr.ReadLine())))
                {
                    string[] now = line.Trim().Split(',');
                    now[0] = "0";
                    list.Add(now);
                }
            }
            list = RandomSort(list);
            using (FileStream fs = new FileStream(root + outputname, FileMode.Create, FileAccess.Write))
            {
                StreamWriter sw = new StreamWriter(fs);
                sw.Write("Class");
                for (int i = 0; i < 1024; i++)
                {
                    sw.Write("\tFP" + (i + 1).ToString());
                }
                sw.Write("\t\n");
                for (int i = 0; i < 1025; i++)
                {
                    sw.Write("d\t");
                }
                sw.Write("\nclass\n");
                foreach (var item in list)
                {
                    for (int i = 0; i < item.Length; i++)
                    {
                        sw.Write(item[i] + "\t");
                    }
                    sw.Write("\n");
                }
                sw.Flush();
            }
        }
        static private List<T> RandomSort<T>(List<T> list)
        {
            var random = new Random();
            var newList = new List<T>();
            foreach (var item in list)
            {
                newList.Insert(random.Next(newList.Count), item);
            }
            return newList;
        }
        static void div(string dataset, int time = 5)
        {
            string dir = @"C:\Users\lenovo\Desktop\毕业论文\result\des\";
            //0:data set
            //1:test or train
            //2:timenow
            //3:fpname
            string outputfilename = @"C:\Users\lenovo\Desktop\毕业论文\result\des\{0}-{1}{2}-{3}";
            List<FingerPrinter> testgroup = new List<FingerPrinter>();
            List<FingerPrinter> traingroup = new List<FingerPrinter>();
            Random r = new Random(1996);
            int index = 1;
            string line;
            FingerPrinter now;
            bool value;
            for (int t = 0; t < time; t++)
            {
                using (FileStream fs = new FileStream(dir + dataset, FileMode.Open, FileAccess.Read))
                {
                    StreamReader sr = new StreamReader(fs);
                    sr.ReadLine();
                    while (!string.IsNullOrEmpty(
                        (line = sr.ReadLine())))
                    {
                        var data = line.Split(',');
                        if (data.Length != 4132)
                            throw new Exception();
                        now = new FingerPrinter();
                        #region fill fp
                        now.name = data[0];
                        for (int i = 0; i < FingerPrinter.FPcount.Length; i++)
                        {
                            for (int j = 0; j < FingerPrinter.FPcount[i]; j++)
                            {
                                now.AllFP[i][j] = data[index];
                                index++;
                            }
                        }
                        #endregion
                        if (r.Next(0, 100) < 30)
                        {
                            testgroup.Add(now);
                        }
                        else
                        {
                            traingroup.Add(now);
                        }
                        index = 1;
                    }
                }
                for (int i = 0; i < FingerPrinter.FPcount.Length; i++)
                {
                    saveGroup(string.Format(outputfilename,
                        dataset, "test", t, FingerPrinter.FPname[i]), testgroup, i);
                    saveGroup(string.Format(outputfilename,
                        dataset, "train", t, FingerPrinter.FPname[i]), traingroup, i);
                }
                testgroup.Clear();
                traingroup.Clear();
            }
        }
        static void saveGroup(string path, List<FingerPrinter> group, int fpindex)
        {
            using (FileStream fs = new FileStream(path, FileMode.Create, FileAccess.Write))
            {
                StreamWriter sw = new StreamWriter(fs);
                foreach (var item in group)
                {
                    sw.Write(item.name);
                    for (int i = 0; i < FingerPrinter.FPcount[fpindex]; i++)
                    {
                        sw.Write("," + item.AllFP[fpindex][i]);
                    }
                    sw.Write("\n");
                }
                sw.Flush();
            }
        }
        static void Main(string[] args)
        {
            toTab("fakertox0-train1-FP", "fakernontox-train1-FP", "train-SubFP.tab");
            toTab("fakertox0-test1-FP", "fakernontox-test1-FP", "test-SubFP.tab");
            //div("fakernontox");
            //div("fakertox0");
        }
    }
}
