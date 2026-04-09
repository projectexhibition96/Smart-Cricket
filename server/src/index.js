import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

import { connectDb } from './lib/db.js';
import { errorHandler } from './middleware/errorHandler.js';
import { authRouter } from './routes/auth.js';
import { aiRouter } from './routes/ai.js';
import { uploadRouter } from './routes/upload.js';

dotenv.config();

const app = express();

/** Allow listed origins; in dev also mirror localhost ↔ 127.0.0.1 (same port) so login works either way. */
function corsOrigin(origin, callback) {
  const raw = process.env.CLIENT_ORIGIN;
  const allowedList = raw
    ? raw
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
    : [];

  if (!origin) {
    return callback(null, true);
  }
  if (allowedList.length === 0) {
    return callback(null, true);
  }
  if (allowedList.includes(origin)) {
    return callback(null, true);
  }
  try {
    const u = new URL(origin);
    const port = u.port || (u.protocol === 'https:' ? '443' : '80');
    const altHost =
      u.hostname === 'localhost' ? '127.0.0.1' : u.hostname === '127.0.0.1' ? 'localhost' : null;
    if (altHost) {
      const alt = `${u.protocol}//${altHost}:${port}`;
      if (allowedList.includes(alt)) {
        return callback(null, true);
      }
    }
  } catch {
    // ignore
  }
  return callback(new Error(`CORS blocked for origin: ${origin}`));
}

app.use(
  helmet({
    // The SPA uses blobs (e.g. three.js textures/GLB resources) and fetches same-origin APIs.
    // Helmet's default CSP is too restrictive and can lead to a blank page in production build.
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        baseUri: ["'self'"],
        frameAncestors: ["'self'"],
        objectSrc: ["'none'"],
        scriptSrc: ["'self'"],
        scriptSrcAttr: ["'none'"],
        styleSrc: ["'self'", 'https:', "'unsafe-inline'"],
        imgSrc: ["'self'", 'data:', 'blob:'],
        fontSrc: ["'self'", 'https:', 'data:'],
        connectSrc: ["'self'"],
        workerSrc: ["'self'", 'blob:'],
        mediaSrc: ["'self'", 'blob:', 'data:'],
        upgradeInsecureRequests: [],
      },
    },
  }),
);
app.use(
  cors({
    origin: corsOrigin,
    credentials: true,
  }),
);
app.use(express.json({ limit: '2mb' }));
app.use(morgan('dev'));

app.get('/api/health', (_req, res) => res.json({ ok: true }));

app.use('/api/auth', authRouter);
app.use('/api/ai', aiRouter);
app.use('/api/upload', uploadRouter);

/**
 * Optional: serve the built Vite app from this same origin (production or single-port demos).
 * Dev workflow: still use `npm run dev` in client/ (port 5173) — Express does not serve the SPA unless built.
 */
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const clientDist = path.resolve(__dirname, '../../client/dist');
// If a client build exists, serve it by default so the app works on a single port.
// Opt out by setting SERVE_CLIENT=0.
const serveClient =
  process.env.SERVE_CLIENT !== '0' && fs.existsSync(path.join(clientDist, 'index.html'));
if (serveClient) {
  app.use(express.static(clientDist));
  // Express/router (newer path-to-regexp) does not accept '*' as a path pattern.
  // Use a regex catch-all instead.
  app.get(/.*/, (req, res, next) => {
    if (req.path.startsWith('/api')) return next();
    res.sendFile(path.join(clientDist, 'index.html'), (err) => (err ? next(err) : undefined));
  });
}

app.use(errorHandler);

// Default 5050: macOS often binds AirPlay to 5000, which breaks the API.
const port = Number(process.env.PORT || 5050);
await connectDb(process.env.MONGODB_URI);
app.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`Smart Cricket API listening on :${port}`);
});

