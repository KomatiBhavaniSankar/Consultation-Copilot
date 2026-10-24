// Plain object on purpose: the server reuses it for the CDN fallback when static/tailwind.css has not been built.
module.exports = {
  content: ["./index.html"],
  theme: {
    extend: {
      colors: {
        ink: "#231f20",
        paper: "#f5efe6",
        sand: "#faf6f0",
        line: "#ded3c2",
        mute: "#6e6055",
        alarm: { DEFAULT: "#a6192e", soft: "#faebed" },
        amber: { DEFAULT: "#9c5700", soft: "#fdf1db" },
        ok: { DEFAULT: "#2d6a4f", soft: "#e7f3ec" },
        // changes with the selected mode (see body[data-mode] in input.css)
        accent: { DEFAULT: "rgb(var(--accent) / <alpha-value>)", soft: "rgb(var(--accent-soft) / <alpha-value>)" },
      },
      fontFamily: {
        sans: ['"Atkinson Hyperlegible"', "system-ui", "-apple-system", '"Segoe UI"', "Roboto", "sans-serif"],
      },
    },
  },
};