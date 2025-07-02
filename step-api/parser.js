import occ from 'node-occ'

export function parseStepFile(filepath) {
  return new Promise((resolve, reject) => {
    occ.readSTEP(filepath, (err, shapes) => {
      if (err) return reject(err)

      const faces = []
      const edges = []
      const vertices = []

      for (const shape of shapes) {
        for (const face of shape.faces()) {
          const mesh = face.tessellate({ deflection: 0.01 })
          faces.push({
            vertices: mesh.points.map(p => [p.x, p.y, p.z]).flat(),
            indices: mesh.triangles.flat()
          })
        }

        for (const edge of shape.edges()) {
          const points = edge.discretize(30)
          edges.push(points.map(p => [p.x, p.y, p.z]))
        }

        for (const vertex of shape.vertices()) {
          const p = vertex.point
          vertices.push([p.x, p.y, p.z])
        }
      }

      resolve({ faces, edges, vertices })
    })
  })
}