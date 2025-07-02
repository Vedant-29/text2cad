import express from 'express'
import multer from 'multer'
import path from 'path'
import { fileURLToPath } from 'url'
import { parseStepFile } from './parser.js'

const app = express()
const port = process.env.PORT || 3000

// Setup __dirname in ES module
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// File upload setup
const upload = multer({ dest: path.join(__dirname, 'uploads') })

app.post('/parse-step', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No STEP file uploaded' })

    const result = await parseStepFile(req.file.path)
    res.json(result)
  } catch (err) {
    console.error('Parse error:', err)
    res.status(500).json({ error: 'Failed to parse STEP file' })
  }
})

app.listen(port, () => {
  console.log(`STEP parser running on http://localhost:${port}`)
})