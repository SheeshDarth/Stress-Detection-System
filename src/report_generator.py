"""
Standalone module to generate HTML reports from session CSV logs.
"""
import os
import csv
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Stress Detection Session Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
        .stat-card { background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2980b9; margin-top: 10px; }
        .chart-container { position: relative; height: 300px; width: 100%; margin-bottom: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Session Review: {date_str}</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div>Total Duration</div>
                <div class="stat-value">{duration:.1f} sec</div>
            </div>
            <div class="stat-card">
                <div>Time Stressed</div>
                <div class="stat-value">{stressed_pct}%</div>
            </div>
            <div class="stat-card">
                <div>Avg Heart Rate</div>
                <div class="stat-value">{avg_bpm} BPM</div>
            </div>
            <div class="stat-card">
                <div>Max Heart Rate</div>
                <div class="stat-value">{max_bpm} BPM</div>
            </div>
        </div>

        <h2>Heart Rate over Time</h2>
        <div class="chart-container">
            <canvas id="bpmChart"></canvas>
        </div>
        
        <h2>Stress Timeline</h2>
        <div class="chart-container">
            <canvas id="stressChart"></canvas>
        </div>
    </div>

    <script>
        const timestamps = {labels};
        const bpmData = {bpm_data};
        const stressData = {stress_data};
        
        new Chart(document.getElementById('bpmChart'), {
            type: 'line',
            data: {
                labels: timestamps,
                datasets: [{
                    label: 'BPM',
                    data: bpmData,
                    borderColor: '#e74c3c',
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        new Chart(document.getElementById('stressChart'), {
            type: 'line',
            data: {
                labels: timestamps,
                datasets: [{
                    label: 'Stress Confidence',
                    data: stressData,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.2)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100 } } }
        });
    </script>
</body>
</html>
"""

def generate_report(csv_filepath: str) -> str:
    if not os.path.exists(csv_filepath):
        return ""
        
    bpm_vals, conf_vals, timestamps = [], [], []
    stressed_count = 0
    total_count = 0
    
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % 5 != 0: continue # Subsample for charts
            try:
                bpm = float(row['bpm'])
                conf = float(row['confidence'])
                label = row['stress_label']
                if bpm > 0:
                    bpm_vals.append(bpm)
                    conf_vals.append(conf * 100 if label == "Stressed" else (1-conf)*100)
                    timestamps.append(i)
                    total_count += 1
                    if label == "Stressed":
                        stressed_count += 1
            except:
                pass
                
    if not bpm_vals: return ""

    avg_bpm = sum(bpm_vals) / len(bpm_vals)
    max_bpm = max(bpm_vals)
    stressed_pct = int((stressed_count / total_count) * 100) if total_count else 0
    duration = total_count * 5 * 0.033 * 5 # Approximation based on ~10fps csv write rate

    html_content = HTML_TEMPLATE.format(
        date_str=datetime.now().strftime("%B %d, %Y - %H:%M"),
        duration=duration,
        stressed_pct=stressed_pct,
        avg_bpm=int(avg_bpm),
        max_bpm=int(max_bpm),
        labels=timestamps,
        bpm_data=bpm_vals,
        stress_data=conf_vals
    )
    
    out_path = csv_filepath.replace(".csv", "_report.html")
    with open(out_path, 'w') as f:
        f.write(html_content)
        
    return out_path
