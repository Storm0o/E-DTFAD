document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ──────────────────────────────────
    const uploadZone = document.getElementById('uploadZone');
    const imageInput = document.getElementById('imageInput');
    const uploadPreview = document.getElementById('uploadPreview');
    const previewImg = document.getElementById('previewImg');
    const removeBtn = document.getElementById('removeBtn');
    const processBtn = document.getElementById('processBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultsSection = document.getElementById('resultsSection');

    const noiseLevel = document.getElementById('noiseLevel');
    const noiseLevelVal = document.getElementById('noiseLevelVal');
    const nIter = document.getElementById('nIter');
    const nIterVal = document.getElementById('nIterVal');
    const dt = document.getElementById('dt');
    const dtVal = document.getElementById('dtVal');

    let selectedFile = null;

    // Range updates
    noiseLevel.addEventListener('input', () => noiseLevelVal.textContent = parseFloat(noiseLevel.value).toFixed(2));
    nIter.addEventListener('input', () => nIterVal.textContent = nIter.value);
    dt.addEventListener('input', () => dtVal.textContent = parseFloat(dt.value).toFixed(2));

    // Upload
    uploadZone.addEventListener('click', () => imageInput.click());
    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', e => {
        e.preventDefault(); uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    imageInput.addEventListener('change', () => { if (imageInput.files.length) handleFile(imageInput.files[0]); });

    removeBtn.addEventListener('click', () => {
        selectedFile = null;
        uploadZone.style.display = '';
        uploadPreview.style.display = 'none';
        processBtn.disabled = true;
        imageInput.value = '';
    });

    function handleFile(f) {
        if (!f.type.startsWith('image/')) return;
        selectedFile = f;
        const r = new FileReader();
        r.onload = e => {
            previewImg.src = e.target.result;
            uploadZone.style.display = 'none';
            uploadPreview.style.display = 'flex';
            processBtn.disabled = false;
        };
        r.readAsDataURL(f);
    }

    // Process
    processBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        loadingOverlay.style.display = 'flex';
        resultsSection.style.display = 'none';

        const fd = new FormData();
        fd.append('image', selectedFile);
        fd.append('noise_type', document.getElementById('noiseType').value);
        fd.append('noise_level', noiseLevel.value);
        fd.append('n_iter', nIter.value);
        fd.append('dt', dt.value);

        try {
            const res = await fetch('/process', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.error) { alert(data.error); return; }
            render(data);
        } catch (e) { alert('Failed: ' + e.message); }
        finally { loadingOverlay.style.display = 'none'; }
    });

    function render(data) {
        resultsSection.style.display = 'block';
        renderParams(data.params);
        renderImages(data.images);
        renderMetrics(data.metrics_original, data.metrics_upgraded);
        renderVerdict(data.metrics_original, data.metrics_upgraded);
        setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth' }), 100);
    }

    function renderParams(p) {
        const grid = document.getElementById('paramsGrid');
        const chips = [
            ['sigma', p.estimated_sigma],
            ['k1', p.k1],
            ['k2', p.k2],
            ['alpha', `${p.alpha_min} – ${p.alpha_max}`],
            ['avg alpha', p.alpha_mean],
            ['terms', `${p.terms_min} – ${p.terms_max}`],
            ['size', p.image_size],
        ];
        grid.innerHTML = chips.map(([l, v], i) =>
            `<div class="param-chip" style="animation-delay:${i*30}ms">
                <span class="p-label">${l}</span>
                <span class="p-val">${v}</span>
            </div>`
        ).join('');
    }

    function renderImages(imgs) {
        const grid = document.getElementById('imagesGrid');
        const tiles = [
            ['clean', 'Clean', '#4ade80'],
            ['noisy', 'Noisy', '#f87171'],
            ['denoised_original', 'Original', '#fb923c'],
            ['denoised_upgraded', 'Upgraded', '#c8a86e'],
        ];
        grid.innerHTML = tiles.map(([k, label, color], i) =>
            `<div class="img-tile" style="animation-delay:${i*80}ms">
                <div class="img-tile-tag">
                    <div class="img-tile-dot" style="background:${color}"></div>
                    <span>${label}</span>
                </div>
                <img src="data:image/png;base64,${imgs[k]}" alt="${label}">
            </div>`
        ).join('');
    }

    function renderMetrics(orig, upg) {
        const container = document.getElementById('metricsComparison');

        function rows(m) {
            return [
                ['PSNR', `${m.psnr_denoised} dB`, m.psnr_gain > 0 ? 'pos' : 'neg'],
                ['Gain', `${m.psnr_gain > 0 ? '+' : ''}${m.psnr_gain} dB`, m.psnr_gain > 0 ? 'pos' : 'neg'],
                ['SSIM', m.ssim_denoised, m.ssim_gain > 0 ? 'pos' : 'neg'],
                ['MSE', m.mse, 'neu'],
                ['Time', `${m.time}s`, 'neu'],
            ].map(([n, v, c]) =>
                `<div class="m-row"><span class="m-name">${n}</span><span class="m-val ${c}">${v}</span></div>`
            ).join('');
        }

        const max = 40;
        const op = Math.min((orig.psnr_denoised / max) * 100, 100);
        const up = Math.min((upg.psnr_denoised / max) * 100, 100);

        container.innerHTML = `
            <div class="m-panel" style="animation-delay:0.1s">
                <div class="m-panel-head">
                    <span class="m-badge orig">Baseline</span>
                    <span class="m-panel-title">Original</span>
                </div>
                ${rows(orig)}
                <div class="psnr-track">
                    <div class="psnr-label">PSNR</div>
                    <div class="psnr-bar-bg"><div class="psnr-fill f-orig" style="width:0%" data-t="${op}"></div></div>
                </div>
            </div>
            <div class="m-panel" style="animation-delay:0.2s">
                <div class="m-panel-head">
                    <span class="m-badge upg">Enhanced</span>
                    <span class="m-panel-title">Upgraded</span>
                </div>
                ${rows(upg)}
                <div class="psnr-track">
                    <div class="psnr-label">PSNR</div>
                    <div class="psnr-bar-bg"><div class="psnr-fill f-upg" style="width:0%" data-t="${up}"></div></div>
                </div>
            </div>`;

        requestAnimationFrame(() => {
            setTimeout(() => {
                document.querySelectorAll('.psnr-fill').forEach(b => b.style.width = b.dataset.t + '%');
            }, 200);
        });
    }

    function renderVerdict(orig, upg) {
        const banner = document.getElementById('winnerBanner');
        const diff = upg.psnr_denoised - orig.psnr_denoised;
        const sd = upg.ssim_denoised - orig.ssim_denoised;

        if (diff > 0) {
            banner.className = 'verdict win-upg';
            banner.innerHTML = `<div class="verdict-icon" style="color:var(--win-green)">+</div>
                <h3 style="color:var(--win-green)">Upgraded Wins</h3>
                <p><strong>+${diff.toFixed(2)} dB</strong> PSNR  |  <strong>+${sd.toFixed(4)}</strong> SSIM</p>`;
        } else {
            banner.className = 'verdict win-orig';
            banner.innerHTML = `<div class="verdict-icon" style="color:var(--warm)">-</div>
                <h3 style="color:var(--warm)">Original Wins</h3>
                <p>Baseline ahead by <strong>${Math.abs(diff).toFixed(2)} dB</strong></p>`;
        }
    }
});
