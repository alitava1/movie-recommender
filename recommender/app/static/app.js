'use strict';

let USER = localStorage.getItem('user') || '';
let USERS = [];
let RATINGS = {};        // movie_id -> {rating, liked, updated_at}
let WATCHLIST = new Set();
let CURRENT_VIEW = 'discover';
let EXPLORE_GENRE = '';
let EXPLORE_OFFSET = 0;
let EXPLORE_TOTAL = 0;
const EXPLORE_LIMIT = 24;
let LAST_RECS = {};      // movie_id -> rec object (for component breakdown in modal)

// Archival muted palettes keyed by primary genre (no neon, no random stock images).
const GENRE_PALETTES = {
  'action':     ['#2B1C19', '#B8543D'],
  'adventure':  ['#252318', '#C49A45'],
  'animation':  ['#1E2623', '#6EA38C'],
  'children':   ['#24261C', '#9EAB5E'],
  'comedy':     ['#282117', '#D4A253'],
  'crime':      ['#1F1C1E', '#9C4E52'],
  'documentary':['#1B2324', '#6E9599'],
  'drama':      ['#1E1B1A', '#A88C6D'],
  'fantasy':    ['#201E26', '#8E82AD'],
  'film-noir':  ['#151515', '#9E978B'],
  'horror':     ['#211516', '#9C3F44'],
  'musical':    ['#261E22', '#C27D8F'],
  'mystery':    ['#181F26', '#6985A3'],
  'romance':    ['#261A1D', '#BA6E78'],
  'sci-fi':     ['#172026', '#6596A8'],
  'thriller':   ['#211B18', '#B86A42'],
  'war':        ['#202119', '#8A9165'],
  'western':    ['#291E16', '#C47D44'],
};

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add('hidden'), 2600);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json();
  if (j.error) { toast(j.error); throw new Error(j.error); }
  return j;
}

function posterSvg(m) {
  const genres = (m && m.genres) || [];
  const primary = genres[0] || 'drama';
  const [bg, accent] = GENRE_PALETTES[primary] || ['#1E1C1A', '#D4A253'];
  const idStr = '#' + String(m.movie_id).padStart(4, '0');
  const year = m.year ? String(m.year) : 'ARCHIVAL';
  const rawTitle = String(m.title || 'Untitled').replace(/\s*\(\d{4}\)\s*$/, '');
  const words = rawTitle.split(/\s+/);
  const lines = [];
  let cur = '';
  for (const w of words) {
    if ((cur + ' ' + w).trim().length > 14 && cur) {
      lines.push(cur);
      cur = w;
      if (lines.length === 3) break;
    } else {
      cur = (cur + ' ' + w).trim();
    }
  }
  if (cur && lines.length < 4) lines.push(cur);
  const titleSvg = lines.map((ln, idx) =>
    `<text x="20" y="${122 + idx * 24}" fill="#F4EFE6" font-family="Georgia, serif" font-size="18" font-weight="700">${esc(ln)}</text>`
  ).join('');
  const motifShift = (m.movie_id * 17) % 40;
  return `<svg class="poster-svg" viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(m.title)} poster">
    <rect width="200" height="300" fill="${bg}"/>
    <rect x="10" y="10" width="180" height="280" fill="none" stroke="${accent}" stroke-opacity="0.35" stroke-width="1"/>
    <circle cx="${145 - motifShift}" cy="64" r="28" fill="none" stroke="${accent}" stroke-opacity="0.22" stroke-width="1"/>
    <line x1="20" y1="84" x2="180" y2="84" stroke="${accent}" stroke-opacity="0.45" stroke-width="1"/>
    <text x="20" y="34" fill="${accent}" font-family="monospace" font-size="11" letter-spacing="2">${esc(year)}</text>
    <text x="180" y="34" text-anchor="end" fill="#A69E92" font-family="monospace" font-size="10">${esc(idStr)}</text>
    ${titleSvg}
    <line x1="20" y1="246" x2="180" y2="246" stroke="${accent}" stroke-opacity="0.35" stroke-width="1"/>
    <text x="20" y="266" fill="${accent}" font-family="monospace" font-size="10" letter-spacing="1.5">${esc(genres.slice(0, 2).join(' · ').toUpperCase() || 'CATALOG')}</text>
  </svg>`;
}

function watchIcon(saved) {
  return saved
    ? `<svg class="icon-svg" viewBox="0 0 24 24"><path d="M5 3h14v18l-7-5-7 5V3z" fill="currentColor"/></svg>`
    : `<svg class="icon-svg" viewBox="0 0 24 24"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>`;
}

function posterMedia(m) {
  const img = m && m.poster_url
    ? `<img src="${esc(m.poster_url)}" alt="${esc(m.title)} poster" class="poster-img" loading="lazy" onerror="this.remove()">`
    : '';
  return posterSvg(m) + img;
}

function posterCard(m) {
  const saved = WATCHLIST.has(m.movie_id);
  const mine = RATINGS[m.movie_id];
  const div = document.createElement('article');
  div.className = 'poster-card';
  div.tabIndex = 0;
  div.onclick = () => showDetail(m.movie_id);
  div.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); showDetail(m.movie_id); } };
  const ratingBadge = mine
    ? `<span class="mono" style="color:var(--accent)">★ ${mine.rating}</span>`
    : (m.mean_rating ? `<span class="mono">${m.mean_rating.toFixed(1)}</span>` : '');
  div.innerHTML = `
    <div class="poster-frame">
      ${posterMedia(m)}
      <button class="watch-fab ${saved ? 'saved' : ''}" title="${saved ? 'Remove from Watchlist' : 'Add to Watchlist'}" aria-label="Toggle watchlist" onclick="event.stopPropagation(); toggleWatch(${m.movie_id})">
        ${watchIcon(saved)}
      </button>
    </div>
    <div class="poster-meta">
      <div class="poster-title">${esc(m.title)}</div>
      <div class="poster-sub">
        <span>${esc((m.genres || []).slice(0, 2).join(', ') || 'Film')}</span>
        ${ratingBadge}
      </div>
    </div>`;
  return div;
}

function setView(name) {
  CURRENT_VIEW = name;
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + name));
  document.querySelectorAll('.nav-btn, .mnav-btn').forEach(b => b.classList.toggle('active', b.dataset.view === name));
  if (name === 'explore' && !document.getElementById('browseList').children.length) loadBrowse();
  if (name === 'watchlist') renderWatchlist();
  if (name === 'ratings') renderMyRatings();
  if (name === 'foryou') renderForYou();
  window.scrollTo({ top: 0, behavior: 'auto' });
}

async function loadUsers() {
  const j = await api('/api/users');
  USERS = j.user_details || j.users.map(u => ({ user_id: u, name: 'user ' + u }));
  const sel = document.getElementById('user');
  sel.innerHTML = '';
  if (!USERS.length) {
    const c = await api('/api/users', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: 'guest' }) });
    return loadUsers();
  }
  USERS.forEach(u => {
    const o = document.createElement('option');
    o.value = u.user_id;
    o.textContent = u.name;
    sel.appendChild(o);
  });
  if (USER && USERS.some(u => String(u.user_id) === String(USER))) sel.value = USER;
  USER = sel.value;
  localStorage.setItem('user', USER);
}

function switchUser() {
  USER = document.getElementById('user').value;
  localStorage.setItem('user', USER);
  refreshAll();
}

async function createUser() {
  const input = document.getElementById('nu');
  const name = input.value.trim();
  if (!name) return;
  const j = await api('/api/users', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
  USER = j.user_id;
  localStorage.setItem('user', USER);
  input.value = '';
  await loadUsers();
  await refreshAll();
  toast('Account created: ' + name);
}

async function resetUser() {
  if (!USER || !confirm('Delete all ratings, watchlist entries, and preferences for this account?')) return;
  await api('/api/reset?user_id=' + USER, { method: 'POST' });
  await refreshAll();
  toast('Account data cleared');
}

async function loadMine() {
  if (!USER) return;
  const [rj, wj] = await Promise.all([
    api('/api/my_ratings?user_id=' + USER),
    api('/api/watchlist?user_id=' + USER),
  ]);
  RATINGS = {};
  rj.ratings.forEach(r => RATINGS[r.movie_id] = r);
  WATCHLIST = new Set(wj.movie_ids || []);
  document.getElementById('navRateCount').textContent = rj.ratings.length;
  document.getElementById('navWatchCount').textContent = WATCHLIST.size;
  updateAccountSummary();
}

async function loadDiscover() {
  if (!USER) return;
  const j = await api('/api/discover?user_id=' + USER);
  renderGenres(j.genres || []);
  renderHero(j.featured);

  LAST_RECS = {};
  (j.recommendations || []).forEach(r => LAST_RECS[r.movie_id] = r);
  const ratedCount = Object.keys(RATINGS).length;
  const notice = document.getElementById('recColdNotice');
  if (ratedCount === 0) {
    notice.textContent = 'New-user mode: showing popular community favorites. Rate a few films to unlock personalized hybrid recommendations.';
    notice.classList.remove('hidden');
  } else {
    notice.classList.add('hidden');
  }
  renderRecShelf(j.recommendations || []);

  const bSec = document.getElementById('becauseSection');
  const bList = document.getElementById('becauseList');
  bList.innerHTML = '';
  if (j.because_you_liked && j.because_you_liked.movies.length) {
    document.getElementById('becauseTitle').textContent = 'Because you liked ' + j.because_you_liked.anchor.title;
    j.because_you_liked.movies.forEach(m => bList.appendChild(posterCard(m)));
    bSec.classList.remove('hidden');
  } else {
    bSec.classList.add('hidden');
  }

  const popList = document.getElementById('popularList');
  popList.innerHTML = '';
  (j.popular || []).forEach(m => popList.appendChild(posterCard(m)));

  const accSec = document.getElementById('acclaimedSection');
  const accList = document.getElementById('acclaimedList');
  accList.innerHTML = '';
  if (j.highest_rated && j.highest_rated.length) {
    j.highest_rated.forEach(m => accList.appendChild(posterCard(m)));
    accSec.classList.remove('hidden');
  } else {
    accSec.classList.add('hidden');
  }

  const recSec = document.getElementById('recentSection');
  const recList = document.getElementById('recentList');
  recList.innerHTML = '';
  if (j.recent_releases && j.recent_releases.length) {
    j.recent_releases.forEach(m => recList.appendChild(posterCard(m)));
    recSec.classList.remove('hidden');
  } else {
    recSec.classList.add('hidden');
  }

  const metaEl = document.getElementById('metadataSummary');
  if (metaEl && j.metadata_status) {
    const ms = j.metadata_status;
    if (!ms.configured) {
      metaEl.textContent = 'TMDB metadata: not configured (using local MovieLens metadata + archival posters; set TMDB_API_KEY or TMDB_READ_TOKEN to enable real artwork)';
    } else if (ms.reachable === false) {
      metaEl.textContent = `TMDB metadata: configured, but api.themoviedb.org is unreachable on this network (${ms.last_error || 'timeout'}; ${ms.cached_movies || 0} cached; using archival fallback)`;
    } else {
      metaEl.textContent = `TMDB metadata: active (${ms.cached_movies || 0} films cached locally; ${ms.links_loaded || 0} ID links)`;
    }
  }

  const gemsSec = document.getElementById('gemsSection');
  const gemsList = document.getElementById('gemsList');
  gemsList.innerHTML = '';
  if (j.hidden_gems && j.hidden_gems.length) {
    j.hidden_gems.forEach(m => {
      const row = document.createElement('div');
      row.className = 'gem-item';
      row.onclick = () => showDetail(m.movie_id);
      row.innerHTML = `
        <div class="gem-thumb">${posterMedia(m)}</div>
        <div>
          <div class="gem-title">${esc(m.title)}</div>
          <div class="muted">${esc((m.genres || []).join(', '))} · ${m.rating_count} ratings</div>
        </div>
        <div class="gem-score">★ ${m.mean_rating ? m.mean_rating.toFixed(2) : '—'}</div>`;
      gemsList.appendChild(row);
    });
    gemsSec.classList.remove('hidden');
  } else {
    gemsSec.classList.add('hidden');
  }
}

function renderHero(m) {
  const slot = document.getElementById('heroSlot');
  if (!m) { slot.innerHTML = ''; return; }
  const saved = WATCHLIST.has(m.movie_id);
  const expText = m.overview
    ? m.overview
    : (m.explanation ? `Featured pick: ${m.explanation}.` : 'Top community favorite in the local archive.');
  const bgStyle = m.backdrop_url
    ? `style="background-image:linear-gradient(95deg,rgba(18,17,16,0.96) 35%,rgba(18,17,16,0.72)),url('${esc(m.backdrop_url)}');background-size:cover;background-position:center;"`
    : '';
  slot.innerHTML = `
    <div class="hero" ${bgStyle}>
      <div class="hero-poster">${posterMedia(m)}</div>
      <div>
        <span class="kicker">${Object.keys(RATINGS).length ? 'Top Recommendation For You' : 'Featured Community Selection'}</span>
        <h1>${esc(m.title)}</h1>
        <div class="hero-meta">
          ${m.year ? `<span>${m.year}</span>` : ''}
          ${m.runtime ? `<span>${m.runtime} min</span>` : ''}
          <span>${(m.genres || []).map(g => `<span class="tag">${esc(g)}</span>`).join('')}</span>
          ${m.mean_rating ? `<span class="hero-rating">★ ${m.mean_rating.toFixed(2)} <span class="muted">(${m.rating_count} ratings)</span></span>` : ''}
        </div>
        <p class="hero-note">${esc(expText)}</p>
        <div class="hero-actions">
          <button class="btn btn-primary" onclick="showDetail(${m.movie_id})">View details &amp; rate</button>
          <button class="btn btn-secondary" onclick="toggleWatch(${m.movie_id})">${saved ? 'In Watchlist' : '+ Watchlist'}</button>
        </div>
      </div>
    </div>`;
}

function renderRecShelf(recs) {
  const el = document.getElementById('recList');
  el.innerHTML = '';
  recs.slice(0, 6).forEach((r, idx) => {
    const row = document.createElement('article');
    row.className = 'rec-row';
    row.onclick = () => showDetail(r.movie_id);
    const c = r.components;
    const bar = c
      ? `<div class="comp-bar">
           <span>CF <strong>${c.cf !== null && c.cf !== undefined ? c.cf.toFixed(2) : '—'}</strong></span>
           <span>Pop <strong>${c.popularity !== null && c.popularity !== undefined ? c.popularity.toFixed(2) : '—'}</strong></span>
           <span>Cont <strong>${c.content !== null && c.content !== undefined ? c.content.toFixed(2) : '—'}</strong></span>
         </div>`
      : '';
    row.innerHTML = `
      <div class="rec-poster-wrap">${posterMedia(r)}<span class="rec-rank">${String(idx + 1).padStart(2, '0')}</span></div>
      <div class="rec-info">
        <div class="rec-title">${esc(r.title)}</div>
        <div class="rec-sub">${esc((r.genres || []).join(' · '))}${r.mean_rating ? ` · ★ ${r.mean_rating.toFixed(1)}` : ''}</div>
        <div class="rec-exp">${esc(r.explanation)}</div>
        <div class="rec-foot">
          ${bar}
          <span class="mono muted">${r.score !== null ? 'score ' + r.score.toFixed(3) : ''}</span>
        </div>
      </div>`;
    el.appendChild(row);
  });
}

async function renderForYou() {
  if (!USER) return;
  const j = await api('/api/recommendations?user_id=' + USER + '&top_n=20');
  const el = document.getElementById('forYouList');
  el.innerHTML = '';
  (j.recommendations || []).forEach((r, idx) => {
    LAST_RECS[r.movie_id] = r;
    const row = document.createElement('div');
    row.className = 'ledger-row';
    row.onclick = () => showDetail(r.movie_id);
    row.innerHTML = `
      <span class="ledger-num">${String(idx + 1).padStart(2, '0')}</span>
      <div class="ledger-thumb">${posterMedia(r)}</div>
      <div>
        <div class="rec-title">${esc(r.title)}</div>
        <div class="rec-sub">${esc((r.genres || []).join(' · '))} · ${r.rating_count} ratings${r.mean_rating ? ` · ★ ${r.mean_rating.toFixed(2)}` : ''}</div>
        <div class="rec-exp">${esc(r.explanation)}</div>
      </div>
      <div class="ledger-score mono">${r.score !== null ? r.score.toFixed(3) : 'Pop #' + (idx + 1)}</div>`;
    el.appendChild(row);
  });
}

function renderGenres(genres) {
  const bar = document.getElementById('genreBar');
  if (bar.children.length) return;
  const all = [''].concat(genres);
  all.forEach(g => {
    const b = document.createElement('button');
    b.className = 'genre-chip' + (EXPLORE_GENRE === g ? ' active' : '');
    b.textContent = g || 'All Genres';
    b.onclick = () => {
      EXPLORE_GENRE = g;
      EXPLORE_OFFSET = 0;
      document.getElementById('q').value = '';
      document.getElementById('searchStatus').classList.add('hidden');
      document.getElementById('exploreKicker').textContent = g ? 'Genre Filter' : 'Full Catalog';
      document.getElementById('exploreHeading').textContent = g ? g.toUpperCase() : 'Explore Films';
      bar.querySelectorAll('.genre-chip').forEach(c => c.classList.toggle('active', c === b));
      loadBrowse();
    };
    bar.appendChild(b);
  });
}

function explorePreset(sortVal) {
  document.getElementById('exploreSort').value = sortVal;
  EXPLORE_OFFSET = 0;
  EXPLORE_GENRE = '';
  setView('explore');
  loadBrowse();
}

function applyExploreFilters() {
  EXPLORE_OFFSET = 0;
  document.getElementById('q').value = '';
  document.getElementById('searchStatus').classList.add('hidden');
  loadBrowse();
}

function resetExploreFilters() {
  EXPLORE_GENRE = '';
  EXPLORE_OFFSET = 0;
  document.getElementById('exploreSort').value = 'popular';
  document.getElementById('yearMin').value = '';
  document.getElementById('yearMax').value = '';
  document.getElementById('minRating').value = '';
  document.getElementById('q').value = '';
  document.getElementById('searchStatus').classList.add('hidden');
  document.getElementById('exploreKicker').textContent = 'Full Catalog';
  document.getElementById('exploreHeading').textContent = 'Explore Films';
  document.querySelectorAll('#genreBar .genre-chip').forEach((c, i) => c.classList.toggle('active', i === 0));
  loadBrowse();
}

async function loadBrowse() {
  const sort = document.getElementById('exploreSort')?.value || 'popular';
  const ymin = document.getElementById('yearMin')?.value || '';
  const ymax = document.getElementById('yearMax')?.value || '';
  const mrat = document.getElementById('minRating')?.value || '';
  const loading = document.getElementById('exploreLoading');
  loading.classList.remove('hidden');
  let url = `/api/browse?offset=${EXPLORE_OFFSET}&limit=${EXPLORE_LIMIT}&sort=${encodeURIComponent(sort)}`;
  if (EXPLORE_GENRE) url += '&genre=' + encodeURIComponent(EXPLORE_GENRE);
  if (ymin) url += '&year_min=' + encodeURIComponent(ymin);
  if (ymax) url += '&year_max=' + encodeURIComponent(ymax);
  if (mrat) url += '&min_rating=' + encodeURIComponent(mrat);
  const j = await api(url);
  loading.classList.add('hidden');
  EXPLORE_TOTAL = j.total;
  const grid = document.getElementById('browseList');
  grid.innerHTML = '';
  if (!j.movies.length) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><p>No films match those filters.</p><button class="btn btn-secondary" onclick="resetExploreFilters()">Reset filters</button></div>`;
  } else {
    j.movies.forEach(m => grid.appendChild(posterCard(m)));
  }
  const pag = document.getElementById('paginationBar');
  pag.classList.remove('hidden');
  const page = Math.floor(EXPLORE_OFFSET / EXPLORE_LIMIT) + 1;
  const pages = Math.max(1, Math.ceil(EXPLORE_TOTAL / EXPLORE_LIMIT));
  document.getElementById('pageInfo').textContent = `Page ${page} of ${pages} (${EXPLORE_TOTAL} films)`;
  document.getElementById('prevPage').disabled = EXPLORE_OFFSET <= 0;
  document.getElementById('nextPage').disabled = EXPLORE_OFFSET + EXPLORE_LIMIT >= EXPLORE_TOTAL;
}

function changePage(delta) {
  EXPLORE_OFFSET = Math.max(0, EXPLORE_OFFSET + delta * EXPLORE_LIMIT);
  loadBrowse();
  window.scrollTo({ top: 0, behavior: 'auto' });
}

async function runSearch() {
  const q = document.getElementById('q').value.trim();
  if (!q) { setView('explore'); return loadBrowse(); }
  setView('explore');
  const j = await api('/api/search?q=' + encodeURIComponent(q));
  const status = document.getElementById('searchStatus');
  status.classList.remove('hidden');
  status.innerHTML = `<span>${j.results.length} result(s) for "<strong>${esc(q)}</strong>"</span>
    <button class="btn btn-link" onclick="clearSearch()">Clear search</button>`;
  document.getElementById('exploreKicker').textContent = 'Search Results';
  document.getElementById('exploreHeading').textContent = `"${q}"`;
  document.getElementById('paginationBar').classList.add('hidden');
  const grid = document.getElementById('browseList');
  grid.innerHTML = '';
  if (!j.results.length) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><p>No films matched "${esc(q)}".</p><button class="btn btn-secondary" onclick="clearSearch()">Return to catalog</button></div>`;
    return;
  }
  j.results.forEach(m => grid.appendChild(posterCard(m)));
}

function clearSearch() {
  document.getElementById('q').value = '';
  document.getElementById('searchStatus').classList.add('hidden');
  document.getElementById('exploreKicker').textContent = 'Full Catalog';
  document.getElementById('exploreHeading').textContent = 'Explore Films';
  EXPLORE_OFFSET = 0;
  loadBrowse();
}

async function renderWatchlist() {
  if (!USER) return;
  const j = await api('/api/watchlist?user_id=' + USER);
  const grid = document.getElementById('watchlistGrid');
  const empty = document.getElementById('watchlistEmpty');
  grid.innerHTML = '';
  if (!j.movies.length) { empty.classList.remove('hidden'); return; }
  empty.classList.add('hidden');
  j.movies.forEach(m => grid.appendChild(posterCard(m)));
}

async function renderMyRatings() {
  if (!USER) return;
  const j = await api('/api/my_ratings?user_id=' + USER);
  const list = document.getElementById('ratingsList');
  const empty = document.getElementById('ratingsEmpty');
  const stats = document.getElementById('ratingStats');
  const rows = j.ratings || [];
  const likedCount = rows.filter(r => r.liked === true || r.rating >= 4.0).length;
  const mean = rows.length ? (rows.reduce((a, b) => a + b.rating, 0) / rows.length).toFixed(2) : '—';
  stats.innerHTML = `
    <div class="stat-cell"><div class="stat-val">${rows.length}</div><div class="stat-lbl">Rated Films</div></div>
    <div class="stat-cell"><div class="stat-val">${likedCount}</div><div class="stat-lbl">Highly Rated (≥ 4)</div></div>
    <div class="stat-cell"><div class="stat-val">${mean}</div><div class="stat-lbl">Mean Rating</div></div>
    <div class="stat-cell"><div class="stat-val">${WATCHLIST.size}</div><div class="stat-lbl">Watchlist</div></div>`;

  list.innerHTML = '';
  if (!rows.length) { empty.classList.remove('hidden'); return; }
  empty.classList.add('hidden');
  const details = await Promise.all(rows.map(r => api('/api/movie/' + r.movie_id).catch(() => ({ movie_id: r.movie_id, title: 'Movie ' + r.movie_id, genres: [] }))));
  const detailMap = {};
  details.forEach(m => detailMap[m.movie_id] = m);
  const filterMode = document.getElementById('ratingsFilter')?.value || 'all';
  const sortMode = document.getElementById('ratingsSort')?.value || 'recent';
  let shown = rows.filter(r => {
    if (filterMode === 'liked') return r.liked === true || r.rating >= 4.0;
    if (filterMode === 'disliked') return r.liked === false || r.rating <= 2.0;
    return true;
  });
  if (sortMode === 'highest') shown.sort((a, b) => b.rating - a.rating);
  else if (sortMode === 'lowest') shown.sort((a, b) => a.rating - b.rating);
  else if (sortMode === 'title') shown.sort((a, b) => (detailMap[a.movie_id]?.title || '').localeCompare(detailMap[b.movie_id]?.title || ''));
  if (!shown.length) {
    list.innerHTML = `<div class="empty-state"><p>No ratings match the selected filter.</p></div>`;
    return;
  }
  shown.forEach(r => {
    const m = detailMap[r.movie_id];
    const row = document.createElement('div');
    row.className = 'history-row';
    row.innerHTML = `
      <div class="gem-thumb" style="cursor:pointer" onclick="showDetail(${m.movie_id})">${posterMedia(m)}</div>
      <div style="cursor:pointer" onclick="showDetail(${m.movie_id})">
        <div class="rec-title">${esc(m.title)}</div>
        <div class="muted">${esc((m.genres || []).join(' · '))} · updated ${esc(r.updated_at)}</div>
      </div>
      <div class="history-actions">
        <span class="meta-chip mono">★ ${r.rating}${r.liked === true ? ' · Liked' : r.liked === false ? ' · Disliked' : ''}</span>
        <button class="btn btn-ghost" onclick="showDetail(${m.movie_id})">Edit</button>
        <button class="btn btn-ghost" onclick="clearRating(${m.movie_id})">Remove</button>
      </div>`;
    list.appendChild(row);
  });
}

async function showDetail(mid) {
  const [m, sim] = await Promise.all([
    api('/api/movie/' + mid),
    api('/api/movie/' + mid + '/similar'),
  ]);
  const mine = RATINGS[mid];
  const saved = WATCHLIST.has(mid);
  const rec = LAST_RECS[mid];
  const recNote = rec ? `<div class="notice">${esc(rec.explanation)}</div>` : '';
  const body = document.getElementById('detailBody');
  const ratings = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
  const metaParts = [
    m.rating_count ? `${m.rating_count} MovieLens ratings` : null,
    m.mean_rating ? `community mean ★ ${m.mean_rating.toFixed(2)}` : null,
    m.tmdb_rating ? `TMDB ★ ${m.tmdb_rating}` : null,
    m.runtime ? `${m.runtime} min` : null,
    m.original_language ? m.original_language.toUpperCase() : null,
  ].filter(Boolean);
  body.innerHTML = `
    <div class="modal-grid">
      <div class="modal-poster">${posterMedia(m)}</div>
      <div>
        <span class="kicker">Catalog #${m.movie_id}${m.year ? ' · ' + m.year : ''}</span>
        <h2 id="modalTitle">${esc(m.title)}</h2>
        <div style="margin:10px 0">${(m.genres || []).map(g => `<span class="tag">${esc(g)}</span>`).join('')}</div>
        <p class="muted">${esc(metaParts.join(' · '))}</p>
        ${m.overview ? `<p class="synopsis">${esc(m.overview)}</p>` : ''}
        ${m.director ? `<div class="credits-line"><strong>Director:</strong> ${esc(m.director)}</div>` : ''}
        ${m.cast && m.cast.length ? `<div class="credits-line"><strong>Cast:</strong> ${esc(m.cast.join(', '))}</div>` : ''}
        ${recNote}
        <div style="margin-top:14px">
          <strong style="font-size:13px">Your rating: ${mine ? '★ ' + mine.rating : 'Unrated'}</strong>
          <div class="rate-strip">${ratings.map(v =>
            `<button class="rate-btn ${mine && mine.rating === v ? 'active' : ''}" onclick="rate(${mid}, ${v})">${v}</button>`
          ).join('')}</div>
          <div class="btn-group">
            <button class="btn btn-secondary" onclick="rate(${mid}, 5.0, true)">Like</button>
            <button class="btn btn-ghost" onclick="rate(${mid}, 1.0, false)">Dislike</button>
            <button class="btn btn-ghost" onclick="toggleWatch(${mid}, true)">${saved ? 'In Watchlist' : '+ Watchlist'}</button>
            ${mine ? `<button class="btn btn-danger" onclick="clearRating(${mid})">Clear rating</button>` : ''}
          </div>
        </div>
      </div>
    </div>
    ${sim.similar && sim.similar.length ? `
      <div class="similar-block">
        <h3>Similar films in catalog (TF-IDF content)</h3>
        <div class="poster-rail" id="modalSimilar"></div>
      </div>` : ''}`;
  if (sim.similar && sim.similar.length) {
    const rail = document.getElementById('modalSimilar');
    sim.similar.forEach(s => rail.appendChild(posterCard(s)));
  }
  document.getElementById('detailModal').classList.remove('hidden');
}

function closeModal() {
  document.getElementById('detailModal').classList.add('hidden');
}

async function rate(mid, value, liked) {
  await api('/api/rate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: parseInt(USER), movie_id: mid, rating: value, liked }),
  });
  await loadMine();
  await showDetail(mid);
  await loadDiscover();
  if (CURRENT_VIEW === 'ratings') renderMyRatings();
  if (CURRENT_VIEW === 'foryou') renderForYou();
  toast('Saved rating ★ ' + value);
}

async function clearRating(mid) {
  await api('/api/unrate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: parseInt(USER), movie_id: mid }),
  });
  await loadMine();
  if (!document.getElementById('detailModal').classList.contains('hidden')) await showDetail(mid);
  await loadDiscover();
  if (CURRENT_VIEW === 'ratings') renderMyRatings();
  toast('Rating removed');
}

async function toggleWatch(mid, fromModal) {
  const j = await api('/api/watchlist/toggle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: parseInt(USER), movie_id: mid }),
  });
  await loadMine();
  if (fromModal) await showDetail(mid);
  if (CURRENT_VIEW === 'watchlist') renderWatchlist();
  if (CURRENT_VIEW === 'discover') loadDiscover();
  if (CURRENT_VIEW === 'explore') loadBrowse();
  toast(j.saved ? 'Added to Watchlist' : 'Removed from Watchlist');
}

function updateAccountSummary() {
  const active = USERS.find(u => String(u.user_id) === String(USER));
  document.getElementById('accountSummary').textContent =
    `Active account: ${active ? active.name : 'user ' + USER} (id ${USER}) · ${Object.keys(RATINGS).length} rated · ${WATCHLIST.size} in watchlist`;
}

async function loadProfile() {
  if (!USER) return;
  const j = await api('/api/personality/profile?user_id=' + USER);
  const el = document.getElementById('persStatus');
  if (j.profile) {
    el.textContent = 'Saved self-reported profile (updated ' + j.profile.updated_at + '). Standard recommendations remain unaffected.';
    for (const t of ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']) {
      document.getElementById('p_' + t).value = j.profile[t];
    }
  } else {
    el.textContent = 'No profile saved. Standard recommendations are active.';
  }
}

async function saveProfile() {
  if (!USER) return;
  const body = { user_id: parseInt(USER) };
  for (const t of ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']) {
    body[t] = parseFloat(document.getElementById('p_' + t).value);
  }
  await api('/api/personality/profile', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  await loadProfile();
  toast('Experimental personality profile saved');
}

async function deleteProfile() {
  if (!USER || !confirm('Delete your self-reported personality profile?')) return;
  await api('/api/personality/profile/delete', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_id: parseInt(USER) }),
  });
  document.getElementById('expRecList').innerHTML = '';
  await loadProfile();
  toast('Personality profile deleted');
}

async function expRecs() {
  if (!USER) return;
  const w = document.getElementById('p_weight').value;
  const j = await api('/api/personality/recommendations?user_id=' + USER + '&weight=' + encodeURIComponent(w), {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
  });
  const el = document.getElementById('expRecList');
  el.innerHTML = '';
  const warn = document.createElement('div');
  warn.className = 'notice';
  warn.innerHTML = `<strong>EXPERIMENTAL MODE</strong> — ${esc(j.meta.warning)} (weight ${j.meta.weight}, fit on ${j.meta.profiled_users} opt-in users).`;
  el.appendChild(warn);
  j.recommendations.forEach((r, idx) => {
    const row = document.createElement('div');
    row.className = 'rec-row';
    row.onclick = () => showDetail(r.movie_id);
    row.innerHTML = `
      <div class="rec-poster-wrap">${posterMedia(r)}<span class="rec-rank">${String(idx + 1).padStart(2, '0')}</span></div>
      <div class="rec-info">
        <div class="rec-title">${esc(r.title)}</div>
        <div class="rec-exp">${esc(r.explanation)}</div>
        <div class="rec-foot"><span class="mono muted">score ${r.score}</span></div>
      </div>`;
    el.appendChild(row);
  });
}

async function refreshAll() {
  await loadMine();
  updateAccountSummary();
  await loadDiscover();
  await loadProfile();
  if (CURRENT_VIEW === 'watchlist') renderWatchlist();
  if (CURRENT_VIEW === 'ratings') renderMyRatings();
  if (CURRENT_VIEW === 'foryou') renderForYou();
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
  if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
    e.preventDefault();
    document.getElementById('q').focus();
  }
});

loadUsers().then(refreshAll);
