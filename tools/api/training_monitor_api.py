#!/usr/bin/env python3
"""
Training Monitor API - FastAPI 版本
與 orbit-engine 整合的訓練監控 API

可整合到現有的 Python 後端（orbit-engine）

Usage:
    # 獨立運行（測試）
    uvicorn api.training_monitor_api:app --reload --port 8001

    # 或整合到 orbit-engine 的主 FastAPI app
    from api.training_monitor_api import router
    app.include_router(router, prefix="/api/training")
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import re
from collections import deque
import threading
import time


# ============================================================================
# 數據模型 (Pydantic)
# ============================================================================

class EpisodeData(BaseModel):
    """單個 Episode 的數據"""
    episode: int
    total_episodes: int
    reward_mean: float
    reward_std: float
    handovers_mean: float
    handovers_std: float
    loss: float
    timestamp: Optional[str] = None


class TrainingStatus(BaseModel):
    """訓練狀態"""
    is_running: bool
    current_episode: int
    total_episodes: int
    progress: float  # 0-100
    latest_reward: float
    latest_loss: float
    estimated_time_remaining: Optional[str] = None


class TrainingMetrics(BaseModel):
    """完整的訓練指標"""
    status: TrainingStatus
    episodes: List[EpisodeData]
    summary: dict


# ============================================================================
# 數據監控器
# ============================================================================

class TrainingDataMonitor:
    """訓練數據監控器（後台線程）"""

    def __init__(self, log_file: str, max_points: int = 1000):
        self.log_file = Path(log_file)
        self.max_points = max_points
        self.data = deque(maxlen=max_points)
        self.last_position = 0
        self.last_update = None

        # 正則表達式匹配日誌
        self.pattern = re.compile(
            r'Episode\s+(\d+)/(\d+):\s+'
            r'reward=([-\d.]+)±([-\d.]+),\s+'
            r'handovers=([-\d.]+)±([-\d.]+),\s+'
            r'loss=([-\d.]+|nan|inf)'
        )

        # 初始讀取
        self.update()

        # 啟動後台更新線程
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def _update_loop(self):
        """後台更新循環"""
        while self.running:
            self.update()
            time.sleep(5)  # 每 5 秒更新

    def update(self):
        """更新數據"""
        if not self.log_file.exists():
            return

        with open(self.log_file, 'r') as f:
            f.seek(self.last_position)
            new_lines = f.readlines()
            self.last_position = f.tell()

        for line in new_lines:
            match = self.pattern.search(line)
            if match:
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

                self.data.append({
                    'episode': episode,
                    'total_episodes': total,
                    'reward_mean': reward_mean,
                    'reward_std': reward_std,
                    'handovers_mean': handovers_mean,
                    'handovers_std': handovers_std,
                    'loss': loss,
                    'timestamp': datetime.now().isoformat()
                })

                self.last_update = datetime.now()

    def get_data(self) -> List[dict]:
        """獲取所有數據"""
        return list(self.data)

    def get_latest(self) -> Optional[dict]:
        """獲取最新數據"""
        return self.data[-1] if self.data else None

    def get_status(self) -> dict:
        """獲取訓練狀態"""
        latest = self.get_latest()
        if not latest:
            return {
                'is_running': False,
                'current_episode': 0,
                'total_episodes': 0,
                'progress': 0.0,
                'latest_reward': 0.0,
                'latest_loss': 0.0,
                'estimated_time_remaining': None
            }

        # 檢查是否還在運行（最後更新時間 < 5 分鐘）
        is_running = False
        if self.last_update:
            time_diff = (datetime.now() - self.last_update).total_seconds()
            is_running = time_diff < 300  # 5 分鐘內有更新

        return {
            'is_running': is_running,
            'current_episode': latest['episode'],
            'total_episodes': latest['total_episodes'],
            'progress': (latest['episode'] / latest['total_episodes']) * 100,
            'latest_reward': latest['reward_mean'],
            'latest_loss': latest['loss'],
            'estimated_time_remaining': self._estimate_time_remaining(latest)
        }

    def _estimate_time_remaining(self, latest: dict) -> Optional[str]:
        """估計剩餘時間"""
        if len(self.data) < 2:
            return None

        # 簡單估計：假設每個 episode 時間相同
        # 實際可以用更精確的方法
        remaining_episodes = latest['total_episodes'] - latest['episode']
        # 假設每個 episode 3 分鐘（可根據實際調整）
        remaining_seconds = remaining_episodes * 180

        hours = remaining_seconds // 3600
        minutes = (remaining_seconds % 3600) // 60

        return f"{int(hours)}h {int(minutes)}m"


# ============================================================================
# FastAPI 應用
# ============================================================================

app = FastAPI(
    title="Training Monitor API",
    description="Real-time training metrics API for RL training",
    version="1.0.0"
)

# CORS 設置（允許前端訪問）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境應該限制具體域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局監控器實例
monitor: Optional[TrainingDataMonitor] = None


@app.on_event("startup")
async def startup_event():
    """啟動時初始化監控器"""
    global monitor
    log_file = "training_level5_20min_final.log"  # 可配置
    if Path(log_file).exists():
        monitor = TrainingDataMonitor(log_file)
        print(f"✅ Training monitor initialized: {log_file}")
    else:
        print(f"⚠️  Warning: Training log not found: {log_file}")


@app.get("/")
async def root():
    """API 根路徑"""
    return {
        "name": "Training Monitor API",
        "version": "1.0.0",
        "endpoints": {
            "/status": "Get training status",
            "/metrics": "Get all training metrics",
            "/episodes": "Get episode data",
            "/latest": "Get latest episode data"
        }
    }


@app.get("/status", response_model=TrainingStatus)
async def get_status():
    """獲取訓練狀態"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized")

    return monitor.get_status()


@app.get("/episodes", response_model=List[EpisodeData])
async def get_episodes(limit: int = 1000):
    """獲取 Episode 數據"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized")

    data = monitor.get_data()
    return data[-limit:] if limit > 0 else data


@app.get("/latest", response_model=Optional[EpisodeData])
async def get_latest():
    """獲取最新 Episode 數據"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized")

    latest = monitor.get_latest()
    if not latest:
        return None
    return latest


@app.get("/metrics", response_model=TrainingMetrics)
async def get_metrics():
    """獲取完整的訓練指標"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized")

    status = monitor.get_status()
    episodes = monitor.get_data()

    # 計算摘要統計
    if episodes:
        rewards = [ep['reward_mean'] for ep in episodes]
        losses = [ep['loss'] for ep in episodes if ep['loss'] != float('inf')]

        summary = {
            'total_episodes_recorded': len(episodes),
            'best_reward': max(rewards) if rewards else 0,
            'worst_reward': min(rewards) if rewards else 0,
            'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
            'min_loss': min(losses) if losses else 0,
            'max_loss': max(losses) if losses else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
        }
    else:
        summary = {}

    return {
        'status': status,
        'episodes': episodes,
        'summary': summary
    }


# ============================================================================
# Router（供整合使用）
# ============================================================================

from fastapi import APIRouter

router = APIRouter()

# 將所有路由添加到 router
router.add_api_route("/status", get_status, methods=["GET"])
router.add_api_route("/episodes", get_episodes, methods=["GET"])
router.add_api_route("/latest", get_latest, methods=["GET"])
router.add_api_route("/metrics", get_metrics, methods=["GET"])


# ============================================================================
# 主程序（測試用）
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("="*70)
    print("🚀 Training Monitor API")
    print("="*70)
    print("")
    print("API 文檔: http://localhost:8001/docs")
    print("ReDoc: http://localhost:8001/redoc")
    print("")
    print("測試端點:")
    print("  GET /status    - 訓練狀態")
    print("  GET /episodes  - Episode 數據")
    print("  GET /latest    - 最新數據")
    print("  GET /metrics   - 完整指標")
    print("")
    print("="*70)

    uvicorn.run(app, host="0.0.0.0", port=8001)
