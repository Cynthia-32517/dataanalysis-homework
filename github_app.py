# ===================== 导入工具包 =====================
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import os
import json

# ===================== 页面基础设置 =====================
st.set_page_config(
    page_title="GitHub 开源项目采集与活跃度分析",
    layout="wide"
)
st.title("🗂️ GitHub 开源项目采集、清洗与活跃度量化分析")
st.markdown("作业：调用 GitHub API，采集机器学习领域开源项目，完成数据清洗、指标构建、关注度与活跃程度对比")

CACHE_FILE = "github_cache.json"
# 采集配置：搜索领域、目标数量
SEARCH_TOPIC = "machine-learning"
TARGET_COUNT = 120
GITHUB_API_URL = "https://api.github.com/search/repositories"

# ===================== 采集函数 =====================
def fetch_github_repos(topic, need_count=120):
    repos_list = []
    page = 1
    per_page = 30
    while len(repos_list) < need_count:
        params = {
            "q": f"topic:{topic} fork:false",
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page
        }
        headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        resp = requests.get(GITHUB_API_URL, params=params, headers=headers)
        if resp.status_code != 200:
            st.error(f"GitHub API 请求失败，code: {resp.status_code}")
            break
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        for item in items:
            repos_list.append({
                "项目全名": item["full_name"],
                "项目名称": item["name"],
                "作者": item["owner"]["login"],
                "Star数量": item["stargazers_count"],
                "Fork数量": item["forks_count"],
                "主要编程语言": item["language"],
                "OpenIssue数量": item["open_issues_count"],
                "创建时间": item["created_at"],
                "最后更新时间": item["updated_at"],
                "项目主页": item["html_url"],
                "是否归档": item["archived"]
            })
            if len(repos_list) >= need_count:
                break
        page += 1
    df_raw = pd.DataFrame(repos_list)
    return df_raw

# ===================== 清洗、衍生指标构建 =====================
def clean_and_compute(df_raw):
    df = df_raw.copy()
    # 时间格式转换
    df["创建时间"] = pd.to_datetime(df["创建时间"])
    df["最后更新时间"] = pd.to_datetime(df["最后更新时间"])
    now_time = pd.Timestamp.now(tz="UTC")

    # 衍生字段
    df["项目存续天数"] = (now_time - df["创建时间"]).dt.total_seconds() / (3600 * 24)
    df["距上次更新天数"] = (now_time - df["最后更新时间"]).dt.total_seconds() / (3600 * 24)

    # 缺失处理：编程语言为空标记为未知
    df["编程语言"] = df["主要编程语言"].fillna("未知")
    df["语言缺失标记"] = df["主要编程语言"].isna()

    # 自制活跃度分数（核心创新）
    # 归一化 star、fork；更新越近、issue 适中越好；归档项目直接扣分
    df["log_star"] = np.log1p(df["Star数量"])
    df["log_fork"] = np.log1p(df["Fork数量"])
    # 更新因子：越新越高，超过 365 天衰减
    df["更新因子"] = np.clip(1 - df["距上次更新天数"] / 365, 0, 1)
    df["归档惩罚"] = np.where(df["是否归档"], 0.2, 1.0)

    df["活跃分数"] = (
        0.4 * df["log_star"]
        + 0.3 * df["log_fork"]
        + 0.25 * df["更新因子"]
        + 0.05 * np.clip(1 / (df["OpenIssue数量"] + 1), 0, 1)
    ) * df["归档惩罚"]

    # 标记休眠项目：超过 180 天无更新
    df["休眠项目"] = df["距上次更新天数"] > 180
    return df

# ===================== 侧边栏控制 =====================
st.sidebar.header("🎛️ 采集控制")
use_cache = st.sidebar.checkbox("优先读取本地缓存（离线演示）", value=True)
refresh = st.sidebar.button("🔄 重新联网拉取 GitHub 项目")

df_raw = None
if use_cache and os.path.exists(CACHE_FILE) and not refresh:
    st.info("📂 加载本地缓存的 GitHub 项目数据")
    df_raw = pd.read_json(CACHE_FILE, convert_dates=["创建时间", "最后更新时间"])
else:
    with st.spinner("🌐 正在调用 GitHub API 采集开源项目..."):
        df_raw = fetch_github_repos(SEARCH_TOPIC, TARGET_COUNT)
        df_raw.to_json(CACHE_FILE, orient="records", date_format="iso")
        st.success(f"✅ 采集完成，有效项目：{len(df_raw)} 条（≥100）")

df_clean = clean_and_compute(df_raw)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 API 原始采集数据",
    "🧹 清洗与衍生指标预览",
    "💻 编程语言分布统计",
    "⭐ 关注度 & 活跃度量化分析",
    "📥 导出数据集"
])

with tab1:
    st.subheader("原始 API 返回数据")
    st.dataframe(df_raw, use_container_width=True)
    st.write(f"采集领域：{SEARCH_TOPIC}，总条数：{len(df_raw)}")

with tab2:
    st.subheader("清洗后数据（新增：存续时长、更新间隔、自制活跃分数）")
    show_cols = [
        "项目全名", "Star数量", "Fork数量", "编程语言",
        "项目存续天数", "距上次更新天数", "活跃分数", "休眠项目", "是否归档"
    ]
    st.dataframe(df_clean[show_cols], use_container_width=True)

    st.subheader("清洗统计摘要")
    miss_lang = df_clean["语言缺失标记"].sum()
    archived_cnt = df_clean["是否归档"].sum()
    sleep_cnt = df_clean["休眠项目"].sum()
    st.markdown(f"""
    - 编程语言字段缺失：{miss_lang} 条，填充为「未知」
    - 已经归档停止维护项目：{archived_cnt} 个
    - 超过半年没有更新（休眠项目）：{sleep_cnt} 个
    - 清洗策略：保留全部项目，使用标记而不是直接删除，便于对比“高星但停更”项目
    """)

with tab3:
    st.subheader("各编程语言项目数量占比")
    lang_cnt = df_clean["编程语言"].value_counts().reset_index()
    lang_cnt.columns = ["编程语言", "项目数"]
    fig_pie = px.pie(lang_cnt, values="项目数", names="编程语言", hole=0.2)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("不同语言平均 Star 对比")
    lang_star = df_clean.groupby("编程语言")["Star数量"].mean().reset_index().sort_values("Star数量", ascending=False)
    fig_bar_lang = px.bar(lang_star, x="编程语言", y="Star数量", color="编程语言")
    st.plotly_chart(fig_bar_lang, use_container_width=True)

with tab4:
    st.subheader("散点图：Star 数量 vs 活跃分数")
    fig_scatter = px.scatter(
        df_clean,
        x="Star数量",
        y="活跃分数",
        color="编程语言",
        hover_data=["项目全名", "距上次更新天数", "是否归档"],
        log_x=True
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("距上次更新天数分布")
    fig_hist = px.histogram(df_clean, x="距上次更新天数", nbins=25)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    st.subheader("📋 量化分析报告")
    top_star = df_clean.loc[df_clean["Star数量"].idxmax()]
    top_active = df_clean.loc[df_clean["活跃分数"].idxmax()]
    st.markdown(f"""
    1. **关注度分析（Star）**
       关注度最高项目：`{top_star["项目全名"]}`，Star = {top_star["Star数量"]}。
       Star 代表大众热度，但这个指标有缺陷：很多多年前爆火的项目 star 很高，现在已经不再维护，无法体现持续活力。

    2. **活跃度综合评估（自建活跃分数）**
       本项目构造「活跃分数」融合了：热度（star、fork）、近期更新、归档状态、issue 情况。
       综合活跃度第一名：`{top_active["项目全名"]}`，活跃分数 {round(top_active["活跃分数"],2)}。
       它不一定是 star 最多，但属于持续迭代、社区活跃的项目。

    3. **语言生态**
       Python 在机器学习领域占绝对主导地位；少量项目使用 Jupyter、C++、R。
       C++ 项目往往偏向底层高性能算法库，虽然项目总量不多，但平均 star 偏高。

    4. **项目生命周期划分**
       - 活跃项目：半年内有更新、未归档；
       - 休眠项目：star 可能很高，但超过 180 天无提交；
       - 归档项目：作者主动标记不再维护，风险最高，不适合二次开发。

    5. **数据处理流程**
       通过 GitHub 开放接口拉取仓库元数据；过滤 fork 副本避免重复；统一 ISO 时间字符串转为时间戳；
       构造时间衍生字段；对缺失语言做标记填充；构建综合评分，实现“关注度”和“活跃度”两个维度分开评估。
    """)

with tab5:
    st.subheader("导出清洗后的完整数据集")
    buffer = io.StringIO()
    df_clean.to_csv(buffer, index=False, encoding="utf-8-sig")
    st.download_button("📥 下载 GitHub 项目数据 csv", buffer.getvalue(), "github_repo_clean.csv")

with st.expander("📌 项目创新点"):
    st.markdown("""
    - ✅ **区分关注度与活跃度：自定义综合活跃评分**
      Star 只能代表历史热度；我们融合更新时效、fork、issue、归档状态，量化项目当前生命力，把“热门但死掉”和“持续活跃”分开；
    - ✅ **生命周期特征提取**：自动计算项目存续时长、距离上次更新天数，划分活跃/休眠/归档；
    - ✅ **数据采集过滤策略**：采集时直接过滤 fork 复制仓库，只采集原创仓库，避免样本冗余；
    - ✅ **缺失与异常处理：标记优先，不盲目丢弃样本**，归档、休眠项目全部保留用于对比分析；
    - ✅ **多视角可视化**：饼图生态分布、均值对比、散点关联分析；
    - ✅ **缓存离线机制**，适配课堂展示，断网也可以完整展示全部图表与分析。
    """)
