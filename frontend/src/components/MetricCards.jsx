import React from 'react';
import { Activity, Flame, Zap } from 'lucide-react';

export default function MetricCards({ softFlux, hardFlux, severity, predictedClass }) {
    // Format scientific numbers nicely if they enter as very small raw sensor floats
    const formatFlux = (val) => {
        if (!val || val === 0) return '0.00 W/m²';
        if (typeof val === 'number') {
            return val < 1e-3 ? val.toExponential(2) + ' W/m²' : val.toFixed(2) + ' W/m²';
        }
        return val + ' W/m²';
    };

    const cards = [
        {
            title: 'Predicted Class',
            value: predictedClass || 'Quiet',
            icon: <Flame className="text-orange-400" />,
            desc: 'Core Engine Classification'
        },
        {
            title: 'Severity Risk Score',
            value: typeof severity === 'number' ? severity.toFixed(2) : '0.00',
            icon: <Zap className="text-yellow-400" />,
            desc: 'Consequence-Weighted Matrix'
        },
        {
            title: 'SoLEXS Soft Flux',
            value: formatFlux(softFlux),
            icon: <Activity className="text-solarOrange" />,
            desc: '0.1 - 0.8nm Spectrum Channel'
        },
        {
            title: 'HEL1OS Hard Flux',
            value: formatFlux(hardFlux),
            icon: <Activity className="text-solarBlue" />,
            desc: '0.05 - 0.4nm Spectrum Channel'
        },
    ];

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {cards.map((c, i) => (
                <div key={i} className="bg-slate-900 border border-slate-800 p-4 rounded-xl">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium text-slate-400">{c.title}</span>
                        {c.icon}
                    </div>
                    <div className="text-2xl font-bold tracking-tight text-white">{c.value}</div>
                    <p className="text-xs text-slate-500 mt-1">{c.desc}</p>
                </div>
            ))}
        </div>
    );
}
