// === Full Menagerie Loader (AEF + frontier context, Brazil GRIP4) + S2 RGB ===
// URL fragment params live AFTER the # and are separated by semicolons.
// Example:
// https://code.earthengine.google.com/#title=Risk66;tag=logit66;year=2022;lat=-9.5;lon=-62.5;zoom=9;
//   lo=-10;hi=10;roadKm=15;nfMaxKm=5;s2m1=7;s2m2=9;s2cloud=60;s2Years=2020,2021,2022,2023;
//   b=-4.18;w=...(66 nums A00..A63,dist_to_nonforest_m,dist_to_road_m);
//
// Required: w (66 numbers). Optional: b.
// Adds Sentinel-2 RGB composites for years listed in s2Years (default: year-2..year+1).
// ============================================================================

// Safe param getter (works in Code Editor; degrades gracefully elsewhere).
function getParam(key, def) {
    return (ui && ui.url && ui.url.get) ? ui.url.get(key, def) : def;
  }
  
  // ---- Read params
  var title = getParam('title', 'AEF + frontier menagerie (66D)');
  var tag   = getParam('tag', '');
  var year  = parseInt(getParam('year', '2022'), 10);
  
  var lat   = parseFloat(getParam('lat', '-9.5'));
  var lon   = parseFloat(getParam('lon', '-62.5'));
  var zoom  = parseInt(getParam('zoom', '9'), 10);
  
  var lo    = parseFloat(getParam('lo', '-10'));
  var hi    = parseFloat(getParam('hi', '10'));
  
  // Context features controls (Better not to increase this very much, cause it can bias your results)
  var roadKm  = parseFloat(getParam('roadKm', '15')); // max road search radius in km 
  var nfMaxKm = parseFloat(getParam('nfMaxKm', '5')); // cap dist-to-nonforest for stability (km)
  
  // Sentinel-2 controls (RGB only)
  var s2m1     = parseInt(getParam('s2m1', '7'), 10);      // start month
  var s2m2     = parseInt(getParam('s2m2', '9'), 10);      // end month
  var s2cloud  = parseFloat(getParam('s2cloud', '60'));    // CLOUDY_PIXEL_PERCENTAGE
  var s2YearsStr = String(getParam('s2Years', '') || '');  // "2020,2021,2022,2023"
  
  // Parse S2 years list. Default: [year-2, year-1, year, year+1]
  function parseIntListCSV(s) {
    s = String(s || '').trim();
    if (!s) return null;
    return s.split(',')
      .map(function(x){ return x.trim(); })
      .filter(function(x){ return x.length > 0; })
      .map(function(x){ return parseInt(x, 10); })
      .filter(function(x){ return isFinite(x); });
  }
  var s2Years = parseIntListCSV(s2YearsStr);
  if (s2Years === null) s2Years = [year-2, year-1, year, year+1];
  
  // Parse weights vector
  var wStr  = String(getParam('w', '') || '');
  var w = wStr
    .split(',')
    .map(function(x) { return x.trim(); })
    .filter(function(x) { return x.length > 0; })
    .map(function(x) { return parseFloat(x); });
  
  // Intercept is optional. Detect presence (b=0 is valid).
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
  print('roadKm:', roadKm, 'nfMaxKm:', nfMaxKm);
  print('S2 months:', s2m1 + '-' + s2m2, 'S2 cloud <=', s2cloud, 'S2 years:', s2Years);
  
  if (w.length !== 66) {
    throw new Error("Expected w length 66 (A00..A63, dist_to_nonforest_m, dist_to_road_m). Got: " + w.length);
  }
  
  // ---- ROI: keep it local around map center (distance ops are heavy)
  var roi = ee.Geometry.Point([lon, lat]).buffer(100000).bounds(); // ~100 km box
  Map.setCenter(lon, lat, zoom);
  
  // =================== DATA ===================
  var embIC = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');
  var lcIC  = ee.ImageCollection('MODIS/061/MCD12Q1'); // yearly, 500m
  var s2SR  = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
  
  // =================== HELPERS ===================
  
  // Forest in IGBP classes (strict forest): 1..5
  function isForestIGBP(lcType1) {
    return lcType1.eq(1)
      .or(lcType1.eq(2))
      .or(lcType1.eq(3))
      .or(lcType1.eq(4))
      .or(lcType1.eq(5));
  }
  
  function aefForYear(y, region) {
    return embIC
      .filterDate(y + '-01-01', (y + 1) + '-01-01')
      .mosaic()
      .clip(region);
  }
  
  function modisLcForYear(y, region) {
    return lcIC
      .filter(ee.Filter.calendarRange(y, y, 'year'))
      .first()
      .select('LC_Type1')
      .clip(region);
  }
  
  // S2 cloud mask (moderate; good enough for RGB comparison)
  function maskS2SR(img) {
    var qa = img.select('QA60');
    var cloud  = qa.bitwiseAnd(1 << 10).neq(0);
    var cirrus = qa.bitwiseAnd(1 << 11).neq(0);
  
    // SCL good classes: 4 vegetation, 5 bare, 6 water, 7 unclassified
    var scl = img.select('SCL');
    var good = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7));
  
    var mask = cloud.or(cirrus).not().and(good);
  
    return img.updateMask(mask).divide(10000);
  }
  
  function s2RgbCompositeForYear(y, monthStart, monthEnd, region, maxCloudPct) {
    var start = ee.Date.fromYMD(y, 1, 1);
    var end   = start.advance(1, 'year');
  
    var ic = s2SR
      .filterBounds(region)
      .filterDate(start, end)
      .filter(ee.Filter.calendarRange(monthStart, monthEnd, 'month'))
      .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', maxCloudPct))
      .map(maskS2SR);
  
    print('S2 count ' + y + ' months ' + monthStart + '-' + monthEnd, ic.size());
  
    return ic.median().clip(region);
  }
  
  function addS2RGB(y, shown) {
    var img = s2RgbCompositeForYear(y, s2m1, s2m2, roi, s2cloud);
    var vis = {bands: ['B4','B3','B2'], min: 0.02, max: 0.30};
    Map.addLayer(img, vis, 'S2 RGB ' + y + ' (' + s2m1 + '-' + s2m2 + ')', shown);
  }
  
  // =================== FRONTIER FEATURES ===================
  
  // dist_to_nonforest_m from MODIS forest/nonforest (500m grid)
  var lc = modisLcForYear(year, roi);
  var forest = isForestIGBP(lc).rename('forest').toByte();
  var nonforest = forest.not().rename('nonforest').toByte();
  
  var pix_m = ee.Number(lc.projection().nominalScale());
  var nfMax_m = ee.Number(nfMaxKm).multiply(1000);
  
  var dist_to_nonforest_m = nonforest.selfMask()
    .fastDistanceTransform(256)
    .sqrt()
    .multiply(pix_m)
    .rename('dist_to_nonforest_m')
    .updateMask(forest)
    .clamp(0, nfMax_m);
  
  // dist_to_road_m using GRIP4 Central-South-America roads
  var ROADS_BR = ee.FeatureCollection('projects/sat-io/open-datasets/GRIP4/Central-South-America')
    .filterBounds(roi);
  
  print('Road features in ROI:', ROADS_BR.size());
  
  var dist_to_road_m = ROADS_BR
    .distance({searchRadius: ee.Number(roadKm).multiply(1000)})
    .rename('dist_to_road_m')
    .clip(roi);
  
  // =================== APPLY 66-D LINEAR MODEL ===================
  
  // AEF bands (A00..A63)
  var aef = aefForYear(year, roi);
  var aBands = aef.bandNames();
  print('AEF band count:', aBands.size());
  
  // Build 66-band image in expected order:
  var bandNames66 = aBands.cat(ee.List(['dist_to_nonforest_m', 'dist_to_road_m']));
  var X = aef.addBands([dist_to_nonforest_m, dist_to_road_m]).select(bandNames66);
  
  var wImg = ee.Image.constant(w).rename(bandNames66);
  
  // Score and prob
  var score = X.multiply(wImg).reduce(ee.Reducer.sum());
  if (hasB) score = score.add(b);
  
  var prob = score.multiply(-1).exp().add(1).pow(-1).rename('prob'); // sigmoid
  score = score.updateMask(forest);
  prob  = prob.updateMask(forest);
  // =================== LAYERS ===================
  
  Map.addLayer(
    score,
    {min: lo, max: hi, palette: ['0000ff', 'ffffff', 'ff0000']},
    (tag ? (tag + ' | ') : '') + title + ' (score)',
    true,
    0.9
  );
  
  Map.addLayer(
    prob,
    {min: 0, max: 1},
    (tag ? (tag + ' | ') : '') + title + ' (prob=sigmoid(score))',
    false,
    0.9
  );
  
  if (hasB) {
    var pred = score.gt(0).selfMask();
    Map.addLayer(
      pred,
      {palette: ['00ff00']},
      (tag ? (tag + ' | ') : '') + title + ' (pred: score>0)',
      false,
      0.65
    );
  }
  
  // =================== SENTINEL-2 RGB COMPARISON ===================
  // Show predicted year and years before (RGB only). Default: year-2..year+1.
  // Turn only 1 layer on by default to reduce clutter.
  for (var i = 0; i < s2Years.length; i++) {
    var y = s2Years[i];
    var shown = (y === year || y === (year + 1)); // show year and year+1 by default
    addS2RGB(y, shown);
  }
  