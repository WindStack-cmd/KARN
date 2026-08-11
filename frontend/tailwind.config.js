/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        spaceBg: "#0B0F19",
        spaceCard: "#111827",
        solarOrange: "#F97316",
        solarBlue: "#3B82F6",
      },
    },
  },
  plugins: [],
}
