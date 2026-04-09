import { useRef, useState } from "react"

export default function UploadZone({ onImage }) {
  const inputRef = useRef()
  const [dragging, setDragging] = useState(false)

  const handle = (file) => {
    if (!file || !file.type.startsWith("image/")) return
    onImage(file)
  }

  return (
    <div
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files[0]) }}
      className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all
        ${dragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"}`}
    >
      <input
        ref={inputRef} type="file" accept="image/*" className="hidden"
        onChange={(e) => handle(e.target.files[0])}
      />
      <div className="text-5xl mb-3">🚦</div>
      <p className="text-gray-600 font-medium">Drop a traffic sign image here</p>
      <p className="text-gray-400 text-sm mt-1">or click to browse — JPG, PNG, PPM</p>
    </div>
  )
}
