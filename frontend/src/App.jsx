import { useState } from "react"
import UploadZone from "./components/UploadZone"
import ResultPanel from "./components/ResultPanel"
import { predictSign } from "./api"

export default function App() {
  const [preview, setPreview]   = useState(null)
  const [result,  setResult]    = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error,   setError]     = useState(null)

  const handleImage = async (file) => {
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setLoading(true)
    try {
      const data = await predictSign(file)
      setResult(data)
    } catch (e) {
      setError("Could not reach the backend. Is it running on port 8000?")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">

      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-2xl mx-auto flex items-center gap-3">
          <span className="text-2xl">🚦</span>
          <div>
            <h1 className="text-lg font-semibold text-gray-800">Traffic Sign Classifier</h1>
            <p className="text-xs text-gray-400">CNN · GTSRB · 43 classes · ~95% accuracy</p>
          </div>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-6 py-8">

        <UploadZone onImage={handleImage} />

        {/* Preview */}
        {preview && (
          <div className="mt-6 flex justify-center">
            <img
              src={preview} alt="Preview"
              className="w-48 h-48 object-cover rounded-2xl border border-gray-200"
            />
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="mt-8 text-center">
            <div className="inline-block w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-gray-500 text-sm mt-3">Classifying...</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-6 bg-red-50 border border-red-200 text-red-600 text-sm rounded-xl p-4">
            {error}
          </div>
        )}

        <ResultPanel result={result} />

      </main>
    </div>
  )
}
