import React from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, ShieldCheck, AlertTriangle } from 'lucide-react';

export default function AlertBanner({ alertLevel, message }) {
    const config = {
        'CRITICAL': { bg: 'bg-red-950 border-red-500 text-red-200', icon: <ShieldAlert className="text-red-400 animate-pulse" />, title: 'CRITICAL RED ALERT' },
        'WARNING': { bg: 'bg-amber-950 border-amber-500 text-amber-200', icon: <AlertTriangle className="text-amber-400 animate-bounce" />, title: 'AMBER WARNING' },
        'ADVISORY': { bg: 'bg-blue-950 border-blue-500 text-blue-200', icon: <AlertTriangle className="text-blue-400" />, title: 'SOLAR ADVISORY' },
        'GREEN': { bg: 'bg-slate-900 border-emerald-600 text-emerald-400', icon: <ShieldCheck />, title: 'GREEN STATE' }
    };

    const current = config[alertLevel] || config['GREEN'];

    return (
        <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className={`border p-4 rounded-xl flex items-center gap-4 transition-all duration-300 ${current.bg}`}
        >
            <div className="p-3 bg-black/40 rounded-lg">{current.icon}</div>
            <div>
                <h2 className="font-bold text-lg tracking-wider">{current.title}</h2>
                <p className="text-sm opacity-80">{message || 'Normal background weather activity profiles logged.'}</p>
            </div>
        </motion.div>
    );
}
