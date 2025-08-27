import subprocess
import time
import csv
import os

# 目标可执行文件（替换成你自己的路径/命令）
COMMAND = ["./build/test"]
OUTPUT_CSV = "gpu_memory_log.csv"

def log_gpu_memory(pid, interval=1):
    peak_mem = 0
    records = []

    while True:
        try:
            # 查询 GPU 显存占用
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"]
            ).decode("utf-8").strip().split("\n")

            found = False
            for line in output:
                p, mem = line.split(", ")
                if int(p) == pid:
                    mem = int(mem)
                    peak_mem = max(peak_mem, mem)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    records.append([timestamp, pid, mem, peak_mem])
                    print(f"[{timestamp}] PID {pid} - Current: {mem} MiB, Peak: {peak_mem} MiB")
                    found = True

            # 如果进程不存在了，退出循环
            if not found:
                break

            time.sleep(interval)

        except subprocess.CalledProcessError:
            break

    # 写入 CSV 文件
    header = ["timestamp", "pid", "current_memory(MiB)", "peak_memory(MiB)"]
    write_header = not os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerows(records)

    print(f"\n日志已写入 {OUTPUT_CSV}")
    print(f"进程 {pid} 的峰值显存使用：{peak_mem} MiB")


if __name__ == "__main__":
    # 启动目标程序
    proc = subprocess.Popen(COMMAND)
    print(f"启动进程 PID={proc.pid}")

    # 记录显存
    log_gpu_memory(proc.pid)

    # 等待进程结束
    proc.wait()
