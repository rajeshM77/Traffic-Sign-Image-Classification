export default function ResultPanel({ result }) {
  if (!result) return null
  const { label, confidence, top5, gradcam_b64 } = result

  const barColor = (conf) => {
    if (conf >= 80) return "bg-green-500"
    if (conf >= 50) return "bg-yellow-400"
    return "bg-red-400"
  }

  return (
    <div className="mt-6 space-y-6">

      {/* Main prediction */}
      <div className="bg-white border border-gray-200 rounded-2xl p-6">
        <p className="text-sm text-gray-400 mb-1">Predicted sign</p>
        <h2 className="text-2xl font-semibold text-gray-800">{label}</h2>
        <div className="mt-3 flex items-center gap-3">
          <div className="flex-1 bg-gray-100 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all duration-700 ${barColor(confidence)}`}
              style={{ width: `${confidence}%` }}
            />
          </div>
          <span className="text-sm font-medium text-gray-700 w-14 text-right">
            {confidence.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Top 5 */}
      <div className="bg-white border border-gray-200 rounded-2xl p-6">
        <p className="text-sm font-medium text-gray-500 mb-4">Top 5 predictions</p>
        <div className="space-y-3">
          {top5.map((p, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-xs text-gray-400 w-4">{i + 1}</span>
              <span className="text-sm text-gray-700 flex-1 truncate">{p.label}</span>
              <div className="w-32 bg-gray-100 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${i === 0 ? "bg-blue-500" : "bg-gray-300"}`}
                  style={{ width: `${p.confidence}%` }}
                />
              </div>
              <span className="text-xs text-gray-500 w-12 text-right">
                {p.confidence.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Grad-CAM */}
      {gradcam_b64 && (
        <div className="bg-white border border-gray-200 rounded-2xl p-6">
          <p className="text-sm font-medium text-gray-500 mb-1">Grad-CAM — what the model focused on</p>
          <p className="text-xs text-gray-400 mb-4">Red = high attention · Blue = low attention</p>
          <img
            src={`data:image/png;base64,${gradcam_b64}`}
            alt="Grad-CAM heatmap"
            className="rounded-xl w-48 h-48 object-cover mx-auto"
            style={{ imageRendering: "pixelated" }}
          />
        </div>
      )}

    </div>
  )
}
