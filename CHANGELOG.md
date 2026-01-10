# 變更日誌 (CHANGELOG)

本檔案記錄 Transformer 比較動畫專案的所有變更。

---

## [2026-01-10 15:13] [新增] 專案初始化

### 影響範圍
- `venv/` - Python 虛擬環境目錄
- `CHANGELOG.md` - 本變更日誌檔案

### 詳細描述
1. 建立 Python 虛擬環境 (`python3 -m venv venv`)
2. 建立變更日誌檔案 `CHANGELOG.md`

### 原因說明
- 虛擬環境確保專案依賴隔離，避免與系統 Python 套件衝突
- 變更日誌用於追蹤專案所有改動，便於回溯和協作

### 測試結果
- [x] `venv/` 目錄存在
- [x] `which python` 顯示虛擬環境路徑: `/data/R10946017/lm_visualization/venv/bin/python`
- [x] `pip list` 顯示乾淨的環境 (僅 pip 和 setuptools)

---
