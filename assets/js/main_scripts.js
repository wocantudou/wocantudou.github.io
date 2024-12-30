import particlesConfig from './config/particlesConfig.js';

document.addEventListener("DOMContentLoaded", function() {
    // 确保DOM完全加载后再执行particlesJS初始化
    particlesJS("particles-js", particlesConfig);

    document.addEventListener("keydown", function(event) {
        if (event.key === "b" || event.key === "B") {
            document.querySelector(".relocate-location").textContent = "Bookmark Page";
            document.querySelector(".relocating").style.opacity = "1";
            setTimeout(function() {
                window.location.href = "bookmarks.html";
            }, 1000);
        }
    });
});
