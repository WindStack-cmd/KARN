import React from 'react';

export default function PredictionTable({ logs }) {
    const getBadge = (lvl) => {
        const s = {
            'CRITICAL': 'bg-red-950 text-red-400 border-red-800',
            'WARNING': 'bg-amber-950 text-amber-400 border-amber-800',
            'ADVISORY': 'bg-blue-950 text-blue-400 border-blue-800',
            'GREEN': 'bg-emerald-950 text-emerald-400 border-emerald-800'
        };
        return `border px-2 py-0.5 rounded text-xs font-bold tracking-wider ${s[lvl] || s['GREEN']}`;
    };

    return (
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl overflow-x-auto">
            <h3 className="text-md font-semibold text-slate-300 mb-4 tracking-wider">SUPABASE CLOUD TELEMETRY DEPLOYMENT LOG (LAST 10 RUNS)</h3>
            <table className="w-full text-left border-collapse">
                <thead>
                    <tr className="border-b border-slate-800 text-slate-400 text-sm">
                        <th className="pb-3">Timestamp (UTC)</th>
                        <th className="pb-3">Output Class</th>
                        <th className="pb-3">Risk Index</th>
                        <th className="pb-3">Alert Trigger</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-slate-300 text-sm">
                    {logs.map((l, i) => (
                        <tr key={i} className="hover:bg-slate-800/40 transition-colors">
                            <td className="py-3 font-mono text-xs">{new Date(l.timestamp).toISOString().replace('T', ' ').substring(0, 19)}</td>
                            <td className="py-3 font-semibold">{l.predicted_class}</td>
                            <td className="py-3 font-mono">{l.severity_score?.toFixed(2)}</td>
                            <td className="py-3"><span className={getBadge(l.alert_level)}>{l.alert_level}</span></td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
