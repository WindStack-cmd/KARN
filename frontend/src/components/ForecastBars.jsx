import React from 'react';

export default function ForecastBars({ probs }) {
    const classes = [
        { label: 'Quiet Profile', val: probs?.Quiet || 0, color: 'bg-emerald-500' },
        { label: 'C-Class Probability', val: probs?.C_Class || 0, color: 'bg-blue-500' },
        { label: 'M-Class Probability', val: probs?.M_Class || 0, color: 'bg-amber-500' },
        { label: 'X-Class Probability', val: probs?.X_Class || 0, color: 'bg-red-500' },
    ];

    return (
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-between">
            <h3 className="text-md font-semibold text-slate-300 tracking-wider mb-2">PROBABILITY DISTRIBUTION MATRIX</h3>
            <div className="space-y-4 my-auto">
                {classes.map((c, i) => (
                    <div key={i}>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-slate-400 font-medium">{c.label}</span>
                            <span className="text-white font-bold">{(c.val * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                            <div className={`h-full rounded-full transition-all duration-500 ${c.color}`} style={{ width: `${c.val * 100}%` }} />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
