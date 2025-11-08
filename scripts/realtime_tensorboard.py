#!/usr/bin/env python3
"""
實時 TensorBoard 監控器
將訓練日誌實時轉換為 TensorBoard 格式

Usage:
    # 啟動實時監控
    python scripts/realtime_tensorboard.py training_level5_20min_final.log &

    # 在另一個終端啟動 TensorBoard
    tensorboard --logdir=logs/tensorboard --port=6006

    # 瀏覽器訪問
    http://localhost:6006
"""

import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class RealtimeTensorBoardMonitor:
    """實時 TensorBoard 監控器"""

    def __init__(self, log_file: str, tensorboard_dir: str = 'logs/tensorboard',
                 update_interval: int = 10):
        """
        Args:
            log_file: 訓練日誌檔案路徑
            tensorboard_dir: TensorBoard 輸出目錄
            update_interval: 更新間隔（秒）
        """
        self.log_file = Path(log_file)
        self.update_interval = update_interval

        # 創建 TensorBoard writer
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(f"{tensorboard_dir}/{run_name}")

        # 記錄上次讀取位置
        self.last_position = 0

        # 正則表達式匹配日誌行
        self.pattern = re.compile(
            r'Episode\s+(\d+)/(\d+):\s+'
            r'reward=([-\d.]+)±([-\d.]+),\s+'
            r'handovers=([-\d.]+)±([-\d.]+),\s+'
            r'loss=([-\d.]+|nan|inf)'
        )

        print(f"✅ TensorBoard 監控器已啟動")
        print(f"   日誌檔案: {self.log_file}")
        print(f"   TensorBoard 目錄: {tensorboard_dir}/{run_name}")
        print(f"   更新間隔: {update_interval} 秒")
        print(f"")
        print(f"🚀 啟動 TensorBoard:")
        print(f"   tensorboard --logdir={tensorboard_dir} --port=6006")
        print(f"")
        print(f"🌐 瀏覽器訪問:")
        print(f"   http://localhost:6006")
        print(f"")
        print(f"📊 監控中...")

    def parse_log_line(self, line: str) -> dict:
        """解析日誌行"""
        match = self.pattern.search(line)
        if not match:
            return None

        episode = int(match.group(1))
        total = int(match.group(2))
        reward_mean = float(match.group(3))
        reward_std = float(match.group(4))
        handovers_mean = float(match.group(5))
        handovers_std = float(match.group(6))

        loss_str = match.group(7)
        try:
            loss = float(loss_str)
        except ValueError:
            loss = float('nan') if loss_str == 'nan' else float('inf')

        return {
            'episode': episode,
            'total': total,
            'reward_mean': reward_mean,
            'reward_std': reward_std,
            'handovers_mean': handovers_mean,
            'handovers_std': handovers_std,
            'loss': loss
        }

    def update(self):
        """更新 TensorBoard 數據"""
        if not self.log_file.exists():
            return False

        # 讀取新的日誌內容
        with open(self.log_file, 'r') as f:
            f.seek(self.last_position)
            new_lines = f.readlines()
            self.last_position = f.tell()

        # 解析新行並寫入 TensorBoard
        new_data_count = 0
        for line in new_lines:
            data = self.parse_log_line(line)
            if data:
                episode = data['episode']

                # 寫入各項指標
                self.writer.add_scalar('Training/Reward_Mean',
                                      data['reward_mean'], episode)
                self.writer.add_scalar('Training/Reward_Std',
                                      data['reward_std'], episode)
                self.writer.add_scalar('Training/Loss',
                                      data['loss'], episode)
                self.writer.add_scalar('Training/Handovers_Mean',
                                      data['handovers_mean'], episode)
                self.writer.add_scalar('Training/Handovers_Std',
                                      data['handovers_std'], episode)

                # 進度百分比
                progress = (episode / data['total']) * 100
                self.writer.add_scalar('Training/Progress', progress, episode)

                new_data_count += 1

        if new_data_count > 0:
            self.writer.flush()
            print(f"📊 [{datetime.now().strftime('%H:%M:%S')}] "
                  f"更新 {new_data_count} 個新數據點")

        return True

    def run(self):
        """運行監控循環"""
        try:
            while True:
                self.update()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\n\n⏹️  監控已停止")
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(
        description='實時 TensorBoard 監控器'
    )
    parser.add_argument('log_file', type=str,
                       help='訓練日誌檔案路徑')
    parser.add_argument('--tensorboard-dir', type=str,
                       default='logs/tensorboard',
                       help='TensorBoard 輸出目錄（預設: logs/tensorboard）')
    parser.add_argument('--interval', type=int, default=10,
                       help='更新間隔秒數（預設: 10）')

    args = parser.parse_args()

    monitor = RealtimeTensorBoardMonitor(
        args.log_file,
        args.tensorboard_dir,
        args.interval
    )

    monitor.run()


if __name__ == '__main__':
    main()
