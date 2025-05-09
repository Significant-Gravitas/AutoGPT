const overlay = document.createElement('div');
Object.assign(overlay.style, {
    position: 'fixed',
    zIndex: 999999,
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    background: 'rgba(0, 0, 0, 0.7)',
    color: '#fff',
    fontSize: '24px',
    fontWeight: 'bold',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
});
const textContent = document.createElement('div');
Object.assign(textContent.style, {
    textAlign: 'center',
});
textContent.textContent = 'AutoGPT Analyzing Page';
overlay.appendChild(textContent);
document.body.append(overlay);
document.body.style.overflow = 'hidden';
let dotCount = 0;
setInterval(() => {
    textContent.textContent = 'AutoGPT Analyzing Page' + '.'.repeat(dotCount);
    dotCount = (dotCount + 1) % 4;
}, 1000);
