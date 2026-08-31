// Next.js page implementing drag-and-drop UI, Wavesurfer preview, and upload to backend
'use client'

import React, { useRef, useState, useEffect } from 'react'
import WaveSurfer from 'wavesurfer.js'

export default function MasteringPage() {
  const [file, setFile] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [progressMsg, setProgressMsg] = useState('')
  const [downloadUrl, setDownloadUrl] = useState('')
  const waveformRef = useRef(null)
  const wavesurferRef = useRef(null)

  useEffect(() => {
    if (waveformRef.current && !wavesurferRef.current) {
      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#3a3226',
        progressColor: '#D4A72C',
        cursorColor: '#F2C94C',
        barWidth: 2,
        height: 96,
      })
    }
    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy()
        wavesurferRef.current = null
      }
    }
  }, [])

  function onDrop(e) {
    e.preventDefault()
    const f = e.dataTransfer.files && e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  function onBrowse(e) {
    const f = e.target.files && e.target.files[0]
    if (f) handleFile(f)
  }

  function handleFile(f) {
    setFile(f)
    setDownloadUrl('')
    const reader = new FileReader()
    reader.onload = (ev) => {
      if (wavesurferRef.current) {
        wavesurferRef.current.loadBlob(new Blob([ev.target.result]))
      }
    }
    reader.readAsArrayBuffer(f)
  }

  async function uploadAndMaster(format='wav', target='streaming') {
    if (!file) return
    setProcessing(true)
    setProgressMsg('Uploading...')
    const form = new FormData()
    form.append('file', file, file.name)
    form.append('target', target)
    form.append('format', format)
    form.append('target_lufs', '-10')

    try {
      const res = await fetch('/api/master', { // Next.js app proxy; in dev, proxy to backend
        method: 'POST',
        body: form,
      })
      if (!res.ok) throw new Error('Processing failed')
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setDownloadUrl(url)
      setProgressMsg('Done — ready to download')
    } catch (err) {
      setProgressMsg('Error: ' + err.message)
    }
    setProcessing(false)
  }

  return (
    <div className="max-w-3xl mx-auto p-6 text-gray-100">
      <div
        onDrop={onDrop}
        onDragOver={(e)=>e.preventDefault()}
        className="border-2 border-dashed border-neutral-700 bg-neutral-900 p-10 rounded-lg text-center">
        <h2 className="text-2xl font-semibold">Drop Unmastered Mix Here</h2>
        <p className="mt-2 text-sm text-neutral-400">WAV, MP3, AIFF up to 2GB</p>
        <input id="fileInput" type="file" accept="audio/*" className="hidden" onChange={onBrowse} />
        <div className="mt-6">
          <button className="px-4 py-2 bg-amber-600 text-black rounded" onClick={()=>document.getElementById('fileInput').click()}>Browse Files</button>
        </div>
      </div>

      <div className="mt-6">
        <div ref={waveformRef} />
      </div>

      <div className="mt-4 flex gap-3">
        <button onClick={()=>uploadAndMaster('wav','streaming')} className="px-4 py-2 bg-gold text-black rounded" disabled={processing||!file}>Master (Streaming)</button>
        <button onClick={()=>uploadAndMaster('wav','highres')} className="px-4 py-2 bg-gray-700 text-white rounded" disabled={processing||!file}>Master (High-Res)</button>
        {downloadUrl && <a href={downloadUrl} download className='px-4 py-2 bg-emerald-600 rounded'>Download</a>}
      </div>

      <div className='mt-4 text-sm text-neutral-400'>
        {processing ? <div>{progressMsg}</div> : <div>{progressMsg}</div>}
      </div>
    </div>
  )
}
