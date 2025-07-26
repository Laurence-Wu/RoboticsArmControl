class FocusStatusApp {
        constructor() {
            this.ws = null;
            this.reconnectAttempts = 0;
            this.maxReconnectAttempts = 10;
            this.reconnectDelay = 1000;
            
            // 添加数据缓存用于平滑过渡
            this.lastUpdateTime = 0;
            this.updateThreshold = 100; // 最小更新间隔100ms
            
            this.elements = {
                statusText: document.getElementById('statusText'),
                statusIndicator: document.getElementById('statusIndicator'),
                connectionStatus: document.getElementById('connectionStatus'),
                attentionValue: document.getElementById('attentionValue'),
                engagementValue: document.getElementById('engagementValue'),
                compositeValue: document.getElementById('compositeValue'),
                fullscreenBtn: document.getElementById('fullscreenBtn')
            };
            
            this.init();
        }
        
        init() {
            this.connectWebSocket();
            this.updateConnectionStatus('Connecting...', 'connecting');
            this.setupFullscreenButton();
        }
        
        setupFullscreenButton() {
            this.elements.fullscreenBtn.addEventListener('click', () => {
                this.toggleFullscreen();
            });
            
            // 监听全屏状态变化
            document.addEventListener('fullscreenchange', () => {
                this.updateFullscreenButton();
            });
            
            document.addEventListener('webkitfullscreenchange', () => {
                this.updateFullscreenButton();
            });
            
            document.addEventListener('mozfullscreenchange', () => {
                this.updateFullscreenButton();
            });
        }
        
        toggleFullscreen() {
            if (!document.fullscreenElement && 
                !document.webkitFullscreenElement && 
                !document.mozFullScreenElement) {
                // 进入全屏
                if (document.documentElement.requestFullscreen) {
                    document.documentElement.requestFullscreen();
                } else if (document.documentElement.webkitRequestFullscreen) {
                    document.documentElement.webkitRequestFullscreen();
                } else if (document.documentElement.mozRequestFullScreen) {
                    document.documentElement.mozRequestFullScreen();
                }
            } else {
                // 退出全屏
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                } else if (document.mozCancelFullScreen) {
                    document.mozCancelFullScreen();
                }
            }
        }
        
        updateFullscreenButton() {
            const isFullscreen = document.fullscreenElement || 
                            document.webkitFullscreenElement || 
                            document.mozFullScreenElement;
            
            const svg = this.elements.fullscreenBtn.querySelector('svg');
            
            if (isFullscreen) {
                // 显示退出全屏图标
                svg.innerHTML = '<path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z"/>';
                this.elements.fullscreenBtn.title = 'Exit Fullscreen';
            } else {
                // 显示进入全屏图标
                svg.innerHTML = '<path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>';
                this.elements.fullscreenBtn.title = 'Enter Fullscreen';
            }
        }
        
        connectWebSocket() {
            try {
                // 自动适配WebSocket地址
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.hostname;
                const wsUrl = `${protocol}//${host}:8765`;
                
                console.log('Attempting to connect to:', wsUrl);
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected to:', wsUrl);
                    this.reconnectAttempts = 0;
                    this.updateConnectionStatus('Connected', 'connected');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected from:', wsUrl);
                    this.updateConnectionStatus('Disconnected', 'disconnected');
                    this.attemptReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus('Connection Error', 'disconnected');
                };
                
            } catch (error) {
                console.error('Failed to create WebSocket connection:', error);
                this.updateConnectionStatus('Connection Failed', 'disconnected');
                this.attemptReconnect();
            }
        }
        
        attemptReconnect() {
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                this.updateConnectionStatus(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`, 'connecting');
                
                setTimeout(() => {
                    this.connectWebSocket();
                }, this.reconnectDelay * this.reconnectAttempts);
            } else {
                this.updateConnectionStatus('Connection Failed', 'disconnected');
            }
        }
        
        handleMessage(data) {
            switch (data.type) {
                case 'connection':
                    console.log('Connection established:', data.message);
                    break;
                    
                case 'attention_update':
                    this.updateFocusStatus(data);
                    break;
                    
                default:
                    console.log('Unknown message type:', data);
            }
        }
        
        updateFocusStatus(data) {
            const currentTime = Date.now();
            
            // 限制更新频率，避免过于频繁的DOM操作
            if (currentTime - this.lastUpdateTime < this.updateThreshold) {
                return;
            }
            this.lastUpdateTime = currentTime;
            
            const isFocused = data.focused === 1;
            const statusText = isFocused ? 'Do Not Disturb' : 'Welcome to Disturb';
            
            // 更新状态文本（添加平滑过渡）
            if (this.elements.statusText.textContent !== statusText) {
                this.elements.statusText.style.opacity = '0.7';
                setTimeout(() => {
                    this.elements.statusText.textContent = statusText;
                    this.elements.statusText.className = `status-text ${isFocused ? 'focused' : 'distracted'}`;
                    this.elements.statusText.style.opacity = '1';
                }, 150);
            }
            
            // 更新状态指示器
            this.elements.statusIndicator.className = `status-indicator ${isFocused ? 'focused' : 'distracted'}`;
            
            // 更新指标值（使用动画数字过渡）
            if (data.metrics) {
                this.animateValue(this.elements.attentionValue, data.metrics.attention, 2);
                this.animateValue(this.elements.engagementValue, data.metrics.engagement, 2);
            }
            
            this.animateValue(this.elements.compositeValue, data.composite_score, 3);
            
            // 添加页面标题更新
            document.title = `${statusText} - Focus Status`;
            
            console.log(`Focus status: ${statusText}, Score: ${data.composite_score.toFixed(3)}`);
        }
        
        // 添加数字动画方法
        animateValue(element, targetValue, decimals) {
            const currentValue = parseFloat(element.textContent) || 0;
            const difference = targetValue - currentValue;
            
            if (Math.abs(difference) < 0.01) return; // 变化太小不执行动画
            
            // 添加更新动画类
            element.classList.add('updating');
            setTimeout(() => element.classList.remove('updating'), 500);
            
            const increment = difference / 15; // 15步完成动画
            let currentStep = 0;
            
            const timer = setInterval(() => {
                currentStep++;
                const newValue = currentValue + (increment * currentStep);
                element.textContent = newValue.toFixed(decimals);
                
                if (currentStep >= 15) {
                    clearInterval(timer);
                    element.textContent = targetValue.toFixed(decimals);
                }
            }, 16); // 约60fps
        }
        
        updateConnectionStatus(message, status) {
            this.elements.connectionStatus.textContent = message;
            this.elements.connectionStatus.className = `connection-status ${status}`;
            
            // 如果正在连接，添加加载动画
            if (status === 'connecting') {
                this.elements.connectionStatus.classList.add('loading');
            } else {
                this.elements.connectionStatus.classList.remove('loading');
            }
        }
    }

    // 页面加载完成后初始化应用
    document.addEventListener('DOMContentLoaded', () => {
        new FocusStatusApp();
    });

    // 页面可见性API - 当页面重新获得焦得焦点时尝试重连
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && window.focusApp) {
            if (!window.focusApp.ws || window.focusApp.ws.readyState === WebSocket.CLOSED) {
                window.focusApp.connectWebSocket();
            }
        }
    });