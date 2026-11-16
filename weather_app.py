# ===================== 导入工具包 =====================
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import io
import json
import os

# ===================== 页面基础设置 =====================
st.set_page_config(
    page_title="多城市气象采集与量化趋势分析",
    layout="wide"
)
st.title("🌤️ 多城市历史天气采集与量化趋势分析")
st.markdown("作业：调用公开气象API，完成数据采集、清洗、衍生指标构建、趋势定量分析与可视化")

# ===================== 城市基础信息库 =====================
city_dict = {
    "广州": {"lat": 23.13, "lon": 113.27},
    "深圳": {"lat": 22.54, "lon": 114.06},
    "汕头": {"lat": 23.36, "lon": 116.67},
    "北京": {"lat": 39.90, "lon": 116.40},
    "杭州": {"lat": 30.27, "lon": 120.16},
    "成都": {"lat": 30.67, "lon": 104.07}
}

# 缓存文件，上课兜底用
CACHE_FILE = "weather_cache.json"
# 【重要】Open‑Meteo archive 接口可调取的最大日期
MAX_AVAILABLE_DATE = date(2026, 11, 14)

# ===================== 侧边栏控制面板 =====================
st.sidebar.header("🎛️ 参数设置")
selected_city_names = st.sidebar.multiselect(
    "选择需要分析的城市",
    options=list(city_dict.keys()),
    default=["广州", "深圳", "汕头"]
)

# 默认：以最大可用日期为终点，往前取 30 天
default_end = MAX_AVAILABLE_DATE
default_start = MAX_AVAILABLE_DATE - timedelta(days=29)

start_date = st.sidebar.date_input("起始日期", value=default_start, max_value=MAX_AVAILABLE_DATE)
end_date = st.sidebar.date_input("结束日期", value=default_end, max_value=MAX_AVAILABLE_DATE)

# 多层日期校验
if end_date > MAX_AVAILABLE_DATE:
    st.error(f"❌ 接口限制：历史天气数据最晚仅支持到 {MAX_AVAILABLE_DATE}，请重新选择结束日期！")
    st.stop()
if end_date < start_date:
    st.error("❌ 结束日期不能早于起始日期！")
    st.stop()

day_diff = (end_date - start_date).days + 1
if day_diff < 30:
    st.warning(f"⚠️ 当前一共只有 {day_diff} 天数据！作业要求至少连续 30 天，请调大时间范围！")
    st.stop()

use_cache = st.sidebar.checkbox("优先读取本地缓存（无网演示开启）", value=True)
refresh_api = st.sidebar.button("🔄 强制从 Open-Meteo 重新拉取最新数据")

metric_map = {
    "平均气温(℃)": "temperature_2m_mean",
    "最高气温(℃)": "temperature_2m_max",
    "最低气温(℃)": "temperature_2m_min",
    "昼夜温差(℃)": "temp_diff",
    "相对湿度(%)": "relative_humidity_2m_mean",
    "温湿体感指数 THI": "thi"
}
select_metric_cn = st.sidebar.selectbox("选择绘图指标", list(metric_map.keys()))

# ===================== 工具函数 =====================
def get_season(start: date, end: date):
    spring = {3,4,5}
    summer = {6,7,8}
    autumn = {9,10,11}
    winter = {12,1,2}
    months = {start.month, end.month}
    res = set()
    for m in months:
        if m in spring: res.add("春季")
        elif m in summer: res.add("夏季")
        elif m in autumn: res.add("秋季")
        elif m in winter: res.add("冬季")
    return "、".join(res) if res else "未知季节"

def calc_thi(temp, rh):
    """温湿体感指数 THI，自主构造衍生指标，表征闷热体感"""
    return 0.8 * temp + (rh / 100) * (temp - 14.4) + 46.4

def get_city_weather(lat, lon, s_date, e_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": s_date.strftime("%Y-%m-%d"),
        "end_date": e_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m_mean",
        "timezone": "Asia/Shanghai"
    }
    resp = requests.get(url, params=params)
    raw = resp.json()
    daily = raw["daily"]
    df = pd.DataFrame({
        "日期": pd.to_datetime(daily["time"]),
        "最高气温(℃)": daily["temperature_2m_max"],
        "最低气温(℃)": daily["temperature_2m_min"],
        "平均气温(℃)": daily["temperature_2m_mean"],
        "相对湿度(%)": daily["relative_humidity_2m_mean"]
    })
    df["昼夜温差(℃)"] = df["最高气温(℃)"] - df["最低气温(℃)"]
    df["温湿体感指数 THI"] = calc_thi(df["平均气温(℃)"], df["相对湿度(%)"])
    df = df.dropna(subset=["平均气温(℃)"])
    return df

def fit_trend(series_x, series_y):
    """最小二乘线性拟合，返回斜率：℃/天"""
    if len(series_x) < 3:
        return 0
    z = np.polyfit(series_x, series_y, deg=1)
    return z[0]

# ===================== 加载数据：缓存 / API =====================
df_all = None
if use_cache and os.path.exists(CACHE_FILE) and not refresh_api:
    st.info("📂 正在加载本地缓存数据，适合课堂离线演示")
    df_all = pd.read_json(CACHE_FILE, convert_dates=["日期"])
else:
    with st.spinner("🔍 正在请求 Open-Meteo 气象接口采集数据..."):
        data_list = []
        for city in selected_city_names:
            lat = city_dict[city]["lat"]
            lon = city_dict[city]["lon"]
            df_one = get_city_weather(lat, lon, start_date, end_date)
            df_one["城市"] = city
            # IQR 异常细分：高温异常 / 低温异常
            q1 = df_one["平均气温(℃)"].quantile(0.25)
            q3 = df_one["平均气温(℃)"].quantile(0.75)
            iqr = q3 - q1
            low_bound = q1 - 1.5 * iqr
            high_bound = q3 + 1.5 * iqr
            df_one["低温异常"] = df_one["平均气温(℃)"] < low_bound
            df_one["高温异常"] = df_one["平均气温(℃)"] > high_bound
            df_one["3日平滑"] = df_one["平均气温(℃)"].rolling(window=3, min_periods=1).mean()
            data_list.append(df_one)
        df_all = pd.concat(data_list, ignore_index=True)
        # 写入缓存
        df_all.to_json(CACHE_FILE, orient="records", date_format="iso")
        st.success("✅ 数据采集完成，已保存到本地缓存！")

season_text = get_season(start_date, end_date)

# 预处理：把日期转为数值，用于拟合趋势
df_all["日期数值"] = df_all["日期"].astype("int64") / 1e9 / 86400

# ===================== 分 Tab 展示，上课演示逻辑清晰 =====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 时序变化曲线",
    "📊 分布与异常统计",
    "📉 量化趋势 & 智能分析",
    "📋 原始数据与导出"
])

color_list = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261", "#264653", "#E9C46A"]

with tab1:
    st.subheader(f"{select_metric_cn} 时序变化（{season_text}）")
    fig = go.Figure()
    for idx, city in enumerate(selected_city_names):
        sub = df_all[df_all["城市"] == city]
        fig.add_trace(go.Scatter(
            x=sub["日期"],
            y=sub[select_metric_cn],
            name=f"{city}｜原始序列",
            line=dict(color=color_list[idx], dash="dot")
        ))
        if select_metric_cn == "平均气温(℃)":
            fig.add_trace(go.Scatter(
                x=sub["日期"],
                y=sub["3日平滑"],
                name=f"{city}｜3日平滑",
                line=dict(color=color_list[idx], width=3)
            ))
    fig.update_layout(height=600, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("箱线图：多城市指标分布对比")
    fig_box = px.box(df_all, x="城市", y=select_metric_cn, color="城市")
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("异常天气统计（基于平均气温 IQR）")
    abnormal_summary = []
    for city in selected_city_names:
        sub = df_all[df_all["城市"] == city]
        total = len(sub)
        low_cnt = sub["低温异常"].sum()
        high_cnt = sub["高温异常"].sum()
        abnormal_summary.append({
            "城市": city,
            "总记录数": total,
            "低温异常天数": int(low_cnt),
            "高温异常天数": int(high_cnt),
            "异常占比(%)": round((low_cnt + high_cnt) / total * 100, 2)
        })
    st.dataframe(pd.DataFrame(abnormal_summary), use_container_width=True)

with tab3:
    st.subheader("趋势量化结果（最小二乘线性拟合）")
    trend_result = []
    for city in selected_city_names:
        sub = df_all[df_all["城市"] == city]
        slope = fit_trend(sub["日期数值"], sub["平均气温(℃)"])
        avg_temp = round(sub["平均气温(℃)"].mean(), 2)
        avg_thi = round(sub["温湿体感指数 THI"].mean(), 2)
        avg_diff = round(sub["昼夜温差(℃)"].mean(), 2)
        trend_result.append({
            "城市": city,
            "平均气温(℃)": avg_temp,
            "气温变化速率(℃/天)": round(slope, 4),
            "平均昼夜温差(℃)": avg_diff,
            "平均体感 THI": avg_thi
        })
    trend_df = pd.DataFrame(trend_result)
    st.dataframe(trend_df, use_container_width=True)

    st.divider()
    st.subheader("🧠 量化自动分析报告")
    city_list_str = "、".join(selected_city_names)
    fastest_rise = trend_df.loc[trend_df["气温变化速率(℃/天)"].idxmax()]
    biggest_diff = trend_df.loc[trend_df["平均昼夜温差(℃)"].idxmax()]

    analysis = f"""
> 观测时段：{start_date} ~ {end_date}，属于{season_text}，对比城市：{city_list_str}

1. **气温变化趋势（定量）**
{fastest_rise["城市"]} 的气温上升速度最快，每日变化速率为 **{fastest_rise["气温变化速率(℃/天)"]} ℃/天**；速率为正代表整体升温，负值代表持续降温。
沿海城市（广州、深圳、汕头）受海洋热容量调节，气温变化速率通常更平缓；北方/内陆城市缺少水体缓冲，气温波动、升降速率往往更大。

2. **昼夜波动特征**
{biggest_diff["城市"]} 的平均昼夜温差最大，达到 {biggest_diff["平均昼夜温差(℃)"]} ℃。昼夜温差反映局地天气的稳定性：差值大说明日间辐射增温、夜间散热差距明显；华南沿海海域调节强，昼夜温差普遍偏小，气温时序更加平稳。

3. **体感舒适度（自建 THI 温湿指数）**
温湿体感指数同时融合气温、相对湿度两项要素，用来刻画人体闷热感受，弥补单纯气温无法体现潮湿闷热的缺陷。THI 越高，闷热不适感越强。
"""
    for row in trend_result:
        analysis += f"- {row['城市']}：平均 THI = {row['平均体感 THI']}\n"

    analysis += """
4. **异常天气解读**
本项目将异常划分为高温异常、低温异常两类，单独统计频次，而不是直接删除异常样本。异常占比越高，说明该时段气温偏离常态的事件越多，气象序列稳定性越差；保留极端样本，对天气波动研究更有价值。

5. **数据处理流程说明**
程序完成接口 JSON 解析、日期格式标准化、缺失样本清洗；采用 IQR 四分位法识别极端气温，滑动平均过滤短期随机扰动；基于最小二乘线性拟合提取长期变化速率，从看图定性描述升级为可横向对比的量化分析。
"""
    st.markdown(analysis)

with tab4:
    st.subheader("全部清洗后数据")
    st.dataframe(df_all, use_container_width=True)
    buffer = io.StringIO()
    df_all.to_csv(buffer, index=False, encoding="utf-8-sig")
    st.download_button("📥 下载全部清洗数据集（csv）", buffer.getvalue(), "weather_analysis.csv")

# ===================== 底部说明 =====================
with st.expander("📌 项目创新说明"):
    st.markdown("""
    - ✅ **衍生指标构建：自主计算温湿体感指数 THI**：融合气温、湿度两个气象要素，刻画人体体感闷热程度，突破原始接口仅有单一气象指标的限制。
    - ✅ **趋势定量分析：最小二乘线性拟合**：计算气温逐日变化速率，将主观看图描述的“升温/降温”转化为可以横向对比的量化指标。
    - ✅ **精细化异常识别**：区分极端高温、极端低温异常，统计异常发生频率，评估气象序列稳定性，不盲目剔除极端样本。
    - ✅ **鲁棒展示方案 + 日期边界保护**：增加日期上限拦截，防止超出接口可访问范围；支持在线采集 / 本地缓存离线读取，专门适配课堂演示，规避断网、接口限流。
    - ✅ **多维度对比体系**：时序曲线（动态演变）+ 箱线分布图（静态分布），多角度横向对比华南等城市气象波动特征。
    """)
