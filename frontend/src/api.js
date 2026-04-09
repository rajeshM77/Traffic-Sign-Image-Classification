const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000"

export async function predictSign(file) {
  const form = new FormData()
  form.append("file", file)
  const res = await fetch(`${BASE}/predict`, { method: "POST", body: form })
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}
