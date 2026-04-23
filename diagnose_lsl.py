#!/usr/bin/env python
"""诊断 LSL 流可用性。

运行此脚本查看当前网络上所有可用的 LSL 流及其属性。
用于排查 LabRecorderCLI 无法录制 XDF 的问题。
"""

import sys


def main() -> int:
    try:
        import pylsl
    except ImportError:
        print("ERROR: pylsl 未安装。请运行: pip install pylsl")
        return 1

    print("=" * 60)
    print("LSL 流诊断")
    print("=" * 60)

    # 解析所有可用的 LSL 流
    print("\n正在搜索 LSL 流（超时 5 秒）...\n")
    streams = pylsl.resolve_streams(5.0)

    if not streams:
        print("❌ 未找到任何 LSL 流！")
        print("\n可能的原因：")
        print("  1. OpenBCI GUI 未启动或未开始数据流")
        print("  2. OpenBCI GUI 的 LSL Networking 未启用")
        print("  3. LSL 流名称/类型配置与查询条件不匹配")
        print("\n请在 OpenBCI GUI 中检查：")
        print("  - 确保 LSL Networking 已启用")
        print("  - 确保 'Start' 按钮已按下（数据正在流式传输）")
        return 1

    print(f"✅ 找到 {len(streams)} 个 LSL 流：\n")

    for i, info in enumerate(streams, 1):
        print(f"--- 流 {i} ---")
        print(f"  名称 (name):     {info.name()}")
        print(f"  类型 (type):     {info.type()}")
        print(f"  通道数:          {info.channel_count()}")
        print(f"  采样率:          {info.nominal_srate()} Hz")
        print(f"  数据类型:        {info.channel_format()}")
        print(f"  源 ID:           {info.source_id()}")
        print(f"  版本:            {info.version()}")
        print(f"  UID:             {info.uid()}")
        print()

    # 检查配置中的查询条件
    print("=" * 60)
    print("LabRecorderCLI 查询条件检查")
    print("=" * 60)

    queries = ['type="EEG"', 'type="Markers"']
    print(f"\nconfig_default.yaml 中的查询条件：")
    for q in queries:
        print(f"  - {q}")

    print("\n匹配结果：")
    for q in queries:
        # 提取 type 值
        if "type=" in q:
            expected_type = q.split('"')[1]
            matching = [s for s in streams if s.type() == expected_type]
            if matching:
                print(f"  ✅ {q} → 匹配到 {len(matching)} 个流")
            else:
                actual_types = list(set(s.type() for s in streams))
                print(f"  ❌ {q} → 无匹配")
                print(f"     实际可用的 type: {actual_types}")

    # 建议的修复
    print("\n" + "=" * 60)
    print("建议")
    print("=" * 60)

    eeg_streams = [s for s in streams if "eeg" in s.type().lower()]
    marker_streams = [s for s in streams if "marker" in s.type().lower() or "mark" in s.type().lower()]

    if eeg_streams and marker_streams:
        print("\n✅ 检测到 EEG 和 Marker 流，但 type 名称可能不匹配。")
        print("   建议修改 config_default.yaml 中的 labrecorder_stream_queries：")
        print(f"   - 'type=\"{eeg_streams[0].type()}\"'")
        print(f"   - 'type=\"{marker_streams[0].type()}\"'")
    elif eeg_streams and not marker_streams:
        print("\n⚠️ 检测到 EEG 流，但未检测到 Marker 流。")
        print("   请在 OpenBCI GUI 中启用 Marker Widget 并配置 LSL 输出。")
    elif not eeg_streams and marker_streams:
        print("\n⚠️ 检测到 Marker 流，但未检测到 EEG 流。")
        print("   请在 OpenBCI GUI 中启用 LSL Networking 的 EEG 流输出。")
    else:
        print("\n❌ 未检测到 EEG 或 Marker 相关的流。")
        print("   请检查 OpenBCI GUI 的 LSL 配置。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
