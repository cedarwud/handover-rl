/**
 * Training Monitor Component
 * React 18 + TypeScript 實時訓練監控組件
 *
 * 可整合到 leo-simulator 前端
 *
 * Usage:
 *   import { TrainingMonitor } from './TrainingMonitor';
 *
 *   <TrainingMonitor apiBaseUrl="http://localhost:8001" />
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

// ============================================================================
// TypeScript 類型定義
// ============================================================================

interface EpisodeData {
  episode: number;
  total_episodes: number;
  reward_mean: number;
  reward_std: number;
  handovers_mean: number;
  handovers_std: number;
  loss: number;
  timestamp?: string;
}

interface TrainingStatus {
  is_running: boolean;
  current_episode: number;
  total_episodes: number;
  progress: number;
  latest_reward: number;
  latest_loss: number;
  estimated_time_remaining?: string;
}

interface TrainingMetrics {
  status: TrainingStatus;
  episodes: EpisodeData[];
  summary: {
    total_episodes_recorded: number;
    best_reward: number;
    worst_reward: number;
    avg_reward: number;
    min_loss: number;
    max_loss: number;
    avg_loss: number;
  };
}

// ============================================================================
// Props
// ============================================================================

interface TrainingMonitorProps {
  apiBaseUrl: string;
  refreshInterval?: number; // 毫秒，預設 5000
  showCharts?: boolean;
  className?: string;
}

// ============================================================================
// 主組件
// ============================================================================

export const TrainingMonitor: React.FC<TrainingMonitorProps> = ({
  apiBaseUrl,
  refreshInterval = 5000,
  showCharts = true,
  className = '',
}) => {
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 獲取訓練指標
  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/metrics`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Failed to fetch training metrics:', err);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  // 自動刷新
  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchMetrics, refreshInterval]);

  if (loading) {
    return <div className={`training-monitor-loading ${className}`}>Loading training data...</div>;
  }

  if (error) {
    return (
      <div className={`training-monitor-error ${className}`}>
        <p>Error loading training data: {error}</p>
        <button onClick={fetchMetrics}>Retry</button>
      </div>
    );
  }

  if (!metrics) {
    return <div className={`training-monitor-no-data ${className}`}>No training data available</div>;
  }

  return (
    <div className={`training-monitor ${className}`}>
      {/* 訓練狀態卡片 */}
      <StatusCards status={metrics.status} />

      {/* 圖表區域 */}
      {showCharts && metrics.episodes.length > 0 && (
        <div className="charts-container">
          <RewardChart episodes={metrics.episodes} />
          <LossChart episodes={metrics.episodes} />
          <HandoverChart episodes={metrics.episodes} />
        </div>
      )}

      {/* 摘要統計 */}
      <SummaryStats summary={metrics.summary} />
    </div>
  );
};

// ============================================================================
// 子組件
// ============================================================================

/**
 * 狀態卡片
 */
const StatusCards: React.FC<{ status: TrainingStatus }> = ({ status }) => {
  return (
    <div className="status-cards">
      <div className="status-card">
        <div className="status-label">Status</div>
        <div className={`status-value ${status.is_running ? 'running' : 'stopped'}`}>
          {status.is_running ? '🟢 RUNNING' : '🔴 STOPPED'}
        </div>
      </div>

      <div className="status-card">
        <div className="status-label">Progress</div>
        <div className="status-value">
          {status.current_episode} / {status.total_episodes}
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${status.progress}%` }}
            />
          </div>
          <div className="progress-text">{status.progress.toFixed(1)}%</div>
        </div>
      </div>

      <div className="status-card">
        <div className="status-label">Latest Reward</div>
        <div className="status-value">{status.latest_reward.toFixed(2)}</div>
      </div>

      <div className="status-card">
        <div className="status-label">Latest Loss</div>
        <div className="status-value">{status.latest_loss.toFixed(4)}</div>
      </div>

      {status.estimated_time_remaining && (
        <div className="status-card">
          <div className="status-label">Est. Time Remaining</div>
          <div className="status-value">{status.estimated_time_remaining}</div>
        </div>
      )}
    </div>
  );
};

/**
 * Reward 圖表
 */
const RewardChart: React.FC<{ episodes: EpisodeData[] }> = ({ episodes }) => {
  return (
    <div className="chart">
      <h3>📈 Episode Reward</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={episodes}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: 'Reward', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="reward_mean"
            stroke="#667eea"
            strokeWidth={2}
            dot={{ r: 3 }}
            name="Mean Reward"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

/**
 * Loss 圖表
 */
const LossChart: React.FC<{ episodes: EpisodeData[] }> = ({ episodes }) => {
  return (
    <div className="chart">
      <h3>📉 Training Loss</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={episodes}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#f093fb"
            strokeWidth={2}
            dot={{ r: 3 }}
            name="Loss"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

/**
 * Handover 圖表
 */
const HandoverChart: React.FC<{ episodes: EpisodeData[] }> = ({ episodes }) => {
  return (
    <div className="chart">
      <h3>🔄 Handover Frequency</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={episodes}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: 'Handovers', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="handovers_mean"
            stroke="#4facfe"
            strokeWidth={2}
            dot={{ r: 3 }}
            name="Mean Handovers"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

/**
 * 摘要統計
 */
const SummaryStats: React.FC<{ summary: TrainingMetrics['summary'] }> = ({ summary }) => {
  if (!summary || Object.keys(summary).length === 0) {
    return null;
  }

  return (
    <div className="summary-stats">
      <h3>📊 Training Summary</h3>
      <div className="stats-grid">
        <div className="stat-item">
          <span className="stat-label">Episodes Recorded:</span>
          <span className="stat-value">{summary.total_episodes_recorded}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Best Reward:</span>
          <span className="stat-value">{summary.best_reward.toFixed(2)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Avg Reward:</span>
          <span className="stat-value">{summary.avg_reward.toFixed(2)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Min Loss:</span>
          <span className="stat-value">{summary.min_loss.toFixed(4)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Avg Loss:</span>
          <span className="stat-value">{summary.avg_loss.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// 導出類型（供外部使用）
// ============================================================================

export type {
  EpisodeData,
  TrainingStatus,
  TrainingMetrics,
  TrainingMonitorProps,
};
