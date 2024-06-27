document.addEventListener('contextmenu', function (e) {
    if (e.target.nodeName === 'VIDEO') {
        e.preventDefault();
    }
}, false);