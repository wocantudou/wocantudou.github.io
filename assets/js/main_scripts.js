import particlesConfig from './config/particlesConfig.js';

// 获取用户位置信息
async function getLocation() {
  try {
    const response = await fetch('https://ipapi.co/json/');
    const data = await response.json();
    const city = data.city || '未知';
    const country = data.country_name || '未知';
    const welcomeText = document.getElementById('welcome-text');
    if (welcomeText) {
      welcomeText.textContent = `欢迎来自${city}，${country}的朋友`;
    }
  } catch (error) {
    console.error('获取位置信息失败:', error);
    const welcomeText = document.getElementById('welcome-text');
    if (welcomeText) {
      welcomeText.textContent = '欢迎访问';
    }
  }
}

// 初始化粒子动画
function initParticles() {
    try {
        particlesJS("particles-js", particlesConfig);
    } catch (error) {
        console.error('粒子动画初始化失败:', error);
    }
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

// 主初始化函数
function init() {
    initParticles();
    initMouseFollow();
    initScrollEffects();
    getLocation();
}

// DOM加载完成后执行初始化
document.addEventListener("DOMContentLoaded", init);
