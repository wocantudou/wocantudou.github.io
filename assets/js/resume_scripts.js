function updateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('zh-CN', { timeZoneName: 'short' });
    document.getElementById('time').innerText = timeString;
}

// 初始化页面时间显示
updateTime();

// 每秒更新时间
setInterval(updateTime, 1000);

document.addEventListener('contextmenu', function (e) {
    if (e.target.nodeName === 'VIDEO') {
        e.preventDefault();
    }
}, false);
