import particlesConfig from './config/particlesConfig.js';

// 初始化粒子动画
function initParticles() {
    try {
        particlesJS("particles-js", particlesConfig);
    } catch (error) {
        console.error('粒子动画初始化失败:', error);
    }
}

// 处理键盘事件
function handleKeydown(event) {
    if (event.key === "b" || event.key === "B") {
        try {
            const relocateLocation = document.querySelector(".relocate-location");
            const relocating = document.querySelector(".relocating");
            
            if (relocateLocation && relocating) {
                relocateLocation.textContent = "Bookmark Page";
                relocating.style.opacity = "1";
                
                setTimeout(() => {
                    window.location.href = "bookmarks.html";
                }, 1000);
            }
        } catch (error) {
            console.error('处理键盘事件失败:', error);
        }
    }
}

// 鼠标跟随效果
function initMouseFollow() {
    const cursor = document.createElement('div');
    cursor.classList.add('custom-cursor');
    document.body.appendChild(cursor);

    document.addEventListener('mousemove', (e) => {
        cursor.style.left = `${e.pageX}px`;
        cursor.style.top = `${e.pageY}px`;
    });

    document.addEventListener('mouseleave', () => {
        cursor.style.opacity = '0';
    });

    document.addEventListener('mouseenter', () => {
        cursor.style.opacity = '1';
    });
}

// 页面滚动效果
function initScrollEffects() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
            }
        });
    }, { threshold: 0.3 });

    elements.forEach(element => {
        observer.observe(element);
    });
}

// 初始化事件监听器
function initEventListeners() {
    document.addEventListener("keydown", handleKeydown);
}

// 主初始化函数
function init() {
    initParticles();
    initMouseFollow();
    initScrollEffects();
    initEventListeners();
}

// DOM加载完成后执行初始化
document.addEventListener("DOMContentLoaded", init);
