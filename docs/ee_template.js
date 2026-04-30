// === LGND Menagerie Loader (v0.2) ==========================================
// URL fragment params live AFTER the # and are separated by semicolons.
// Example:
//   ...?code#title=Corn%20test;year=2024;lat=49.88;lon=2.60;zoom=11;
//        lo=-25;hi=25;b=-5.32;w=2.95,-3.13,...;
//
// Params:
//   title : string (optional)
//   tag   : string (optional, for printing / naming)
//   year  : int (default 2024)
//   lat   : float (default 0)
//   lon   : float (default 0)
//   zoom  : int (default 10)
//   lo    : float (default -20)  score viz min
//   hi    : float (default  20)  score viz max
//   w     : comma-separated vector weights (required)
//   b     : intercept (optional). If present, adds pred layer score>0.
// ===========================================================================

// Safe param getter (works in Code Editor; degrades gracefully elsewhere).
function getParam(key, def) {
  return (ui && ui.url && ui.url.get) ? ui.url.get(key, def) : def;
}

// ---- Read params
var title = getParam('title', 'Unnamed specimen');
var tag   = getParam('tag', '');
var year  = parseInt(getParam('year', '2024'), 10);

var lat   = parseFloat(getParam('lat', '0'));
var lon   = parseFloat(getParam('lon', '0'));
var zoom  = parseInt(getParam('zoom', '10'), 10);

var lo    = parseFloat(getParam('lo', '-20'));
var hi    = parseFloat(getParam('hi', '20'));

var wStr  = String(getParam('w', '') || '');
var w = wStr
  .split(',')
  .map(function(x) { return x.trim(); })
  .filter(function(x) { return x.length > 0; })
  .map(function(x) { return parseFloat(x); });

// Intercept is optional. We need to detect presence, not truthiness (b=0 is valid).
var bStr = getParam('b', null);
var hasB = (bStr !== null && bStr !== undefined && String(bStr).trim().length > 0);
var b = hasB ? parseFloat(bStr) : 0;

// ---- Debug prints
print('Title:', title);
if (tag) print('Tag:', tag);
print('Year:', year);
print('Map start:', {lat: lat, lon: lon, zoom: zoom});
print('Score range:', {lo: lo, hi: hi});
print('Vector length:', w.length);
print('Has intercept (b)?', hasB, hasB ? ('b=' + b) : '');

// Guard: if no vector, stop early with a helpful message.
if (w.length === 0) {
  throw new Error("No 'w' vector provided. Pass w=... in the URL fragment after #");
}

// ---- AEF image for the year
var aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate(year + '-01-01', (year + 1) + '-01-01')
  .mosaic();

// Optional sanity info (server-side)
print('AEF band count:', aef.bandNames().size());

// ---- Score: dot(aef, w) (+ b if provided)
var wImg = ee.Image.constant(w).rename(aef.bandNames());
var score = aef
  .multiply(wImg)
  .reduce(ee.Reducer.sum());

if (hasB) {
  score = score.add(b);
}

// ---- Layers
// Score layer (always)
Map.addLayer(
  score,
  {min: lo, max: hi, palette: ['0000ff', 'ffffff', 'ff0000']},
  (tag ? (tag + ' | ') : '') + title + ' (score)',
  true,
  0.9
);

// Pred layer (only if intercept provided)
if (hasB) {
  var pred = score.gt(0).selfMask();
  Map.addLayer(
    pred,
    {palette: ['00ff00']},
    (tag ? (tag + ' | ') : '') + title + ' (pred: score>0)',
    true,
    0.65
  );
}

// ---- Map start position
// If lat/lon were left at 0,0, fall back to centering on something meaningful.
if (isFinite(lat) && isFinite(lon) && !(lat === 0 && lon === 0)) {
  Map.setCenter(lon, lat, zoom);
} else {
  // Fallback: center on the score footprint (can be global-ish depending on how you run it).
  Map.centerObject(score.geometry(), zoom);
}

// Reference URL with a real POME Lagoons model (zoom=15, lat/lon over Riau, Sumatra):
// https://code.earthengine.google.com/54ea5eaf4d04958ef54c4378e7d0e13d?code#title=POME%20Lagoons;year=2024;lat=0.210272;lon=101.2752018;zoom=15;lo=0;hi=0.5;w=-0.0564,...;
