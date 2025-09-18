let currentChatId = null;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        mermaid.initialize({
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose'
        });

        let messageCounter = 0;

        document.addEventListener('DOMContentLoaded', function() {
            const inputAreas = document.querySelectorAll('.input-area');
            inputAreas.forEach(area => {
                area.addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = Math.min(this.scrollHeight, 300) + 'px';
                });
                area.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        if (this.id === 'summarizeInput' || this.id === 'modalSummarizeInput') processSummarization();
                    }
                });
            });

            setupAudioRecording().catch(error => {
                console.error('Failed to setup audio recording:', error);
                document.getElementById('audioStatus').textContent = 'Error: Microphone access denied';
                document.getElementById('startBtn').disabled = true;
            });
        });

        async function setupAudioRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const formData = new FormData();
                    formData.append('audio', audioBlob);

                    document.getElementById('audioStatus').textContent = 'Processing audio...';

                    try {
                        const response = await fetch('/transcribe', {
                            method: 'POST',
                            body: formData
                        });

                        const result = await response.json();
                        if (!response.ok) throw new Error(result.error || 'Transcription failed');

                        if (result.text) {
                            if (result.text.startsWith('Transcription error')) throw new Error(result.text);
                            const inputArea = document.getElementById('summarizeInput');
                            inputArea.innerText = result.text;
                            inputArea.style.height = 'auto';
                            inputArea.style.height = Math.min(inputArea.scrollHeight, 300) + 'px';
                            document.getElementById('audioStatus').textContent = 'Transcription complete!';
                        } else {
                            throw new Error('No transcription text received');
                        }
                    } catch (error) {
                        console.error('Transcription error:', error);
                        document.getElementById('audioStatus').textContent = `Error: ${error.message || 'Failed to transcribe audio'}. Please try again.`;
                    }
                };

                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('audioStatus').textContent = 'Microphone ready. Click "Start" to begin.';
            } catch (err) {
                console.error('Error accessing microphone:', err);
                document.getElementById('audioStatus').textContent = 'Error: Please allow microphone access to use this feature.';
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = true;
                throw err;
            }
        }

        document.getElementById('startBtn').addEventListener('click', () => {
            if (!isRecording && mediaRecorder) {
                audioChunks = [];
                mediaRecorder.start();
                isRecording = true;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('audioStatus').textContent = 'Recording in progress...';
            }
        });

        document.getElementById('stopBtn').addEventListener('click', () => {
            if (isRecording && mediaRecorder) {
                mediaRecorder.stop();
                isRecording = false;
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('audioStatus').textContent = 'Processing transcription...';
            }
        });

        function switchToChatUI() {
            document.getElementById('initialContainer').style.display = 'none';
            document.getElementById('chatContainer').style.display = 'flex';
        }

        function addMessage(content, type) {
            const chatOutput = document.getElementById('chatOutput');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            if (type === 'system') {
                messageCounter++;
                messageDiv.id = `system-message-${messageCounter}`;
                messageDiv.innerHTML = `
                    <div class="message-content">${content}</div>
                    <span class="export-btn" title="Export as PDF"><i class="fas fa-download"></i></span>
                `;
            } else {
                messageDiv.innerHTML = content;
            }
            
            chatOutput.appendChild(messageDiv);
            chatOutput.scrollTop = chatOutput.scrollHeight;
            
            if (content.includes('class="mermaid"')) {
                mermaid.init(undefined, document.querySelectorAll('.mermaid'));
            }
        }

        document.getElementById('fileInput').addEventListener('change', async function(e) {
            handleFileUpload(e, 'fileInfo', 'summarizeInput');
        });

        document.getElementById('modalFileInput').addEventListener('change', async function(e) {
            handleFileUpload(e, 'modalFileInfo', 'modalSummarizeInput');
        });

        async function handleFileUpload(e, infoId, inputId) {
            const file = e.target.files[0];
            if (!file) return;

            const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
            if (!validTypes.includes(file.type)) {
                alert('Please upload a PDF or DOCX file');
                return;
            }

            document.getElementById(infoId).innerHTML = `
                <strong>${file.name}</strong><br>
                ${(file.size / 1024).toFixed(2)} KB<br>
                <span class="loading-text">Processing...</span>
            `;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/extract-text', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) throw new Error('Failed to extract text');
                const result = await response.json();

                if (result.text) {
                    document.getElementById(inputId).innerText = result.text;
                    document.getElementById(inputId).style.height = 'auto';
                    document.getElementById(inputId).style.height = Math.min(document.getElementById(inputId).scrollHeight, 300) + 'px';
                    document.getElementById(infoId).innerHTML = `
                        <strong>${file.name}</strong><br>
                        ${(file.size / 1024).toFixed(2)} KB<br>
                        <span style="color: #7ee787;">✓ Extracted</span>
                    `;
                } else if (result.error) {
                    document.getElementById(infoId).innerHTML = `
                        <strong>${file.name}</strong><br>
                        ${(file.size / 1024).toFixed(2)} KB<br>
                        <span style="color: #f85149;">Error: ${result.error}</span>
                    `;
                }
            } catch (error) {
                console.error('Error:', error);
                document.getElementById(infoId).innerHTML = `
                    <strong>${file.name}</strong><br>
                    ${(file.size / 1024).toFixed(2)} KB<br>
                    <span style="color: #f85149;">Error</span>
                `;
            }
        }

        async function processSummarization() {
            const inputArea = document.getElementById('summarizeInput').innerText.trim() || document.getElementById('modalSummarizeInput').innerText.trim();
            if (!inputArea) {
                alert('Please enter or paste some text to summarize');
                return;
            }

            switchToChatUI();
            const progressBar = document.querySelector('.progress-bar');
            const loading = document.querySelector('.loading');
            
            loading.style.display = 'block';
            progressBar.style.width = '50%';

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: inputArea, is_audio: false, chat_id: currentChatId })
                });

                const result = await response.json();
                currentChatId = result.chat_id;

                addMessage(`
                    <h3>Summary</h3>
                    <div>${result.abstractive_report.replace(/<think>.*?<\/think>/gs, '')}</div>
                `, 'system');
                hideSummarizeModal();
            } catch (error) {
                console.error('Error:', error);
                addMessage('Error processing text. Please try again.', 'system');
            }

            progressBar.style.width = '100%';
            loading.style.display = 'none';
            setTimeout(() => progressBar.style.width = '0', 1000);
        }

        async function sendChatMessage() {
            const chatInput = document.getElementById('chatInput');
            const message = chatInput.value.trim();
            if (!message) return;
            
            addMessage(`<strong>You:</strong> ${message}`, 'user');
            chatInput.value = '';
            
            const progressBar = document.querySelector('.progress-bar');
            const loading = document.querySelector('.loading');
            
            loading.style.display = 'block';
            progressBar.style.width = '50%';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, chat_id: currentChatId })   
                });
                
                const result = await response.json();
                const cleanedResponse = result.response.replace(/<think>.*?<\/think>/gs, '');
                addMessage(`<strong>Assistant:</strong> ${cleanedResponse}`, 'system');
            } catch (error) {
                addMessage(`<strong>Assistant:</strong> Error: Could not get response`, 'system');
            }

            progressBar.style.width = '100%';
            loading.style.display = 'none';
            setTimeout(() => progressBar.style.width = '0', 1000);
        }

        function clearChat() {
            document.getElementById('chatOutput').innerHTML = `
                <div class="message system-message">
                    Start summarizing text, audio, or ask questions here.
                </div>
            `;
            document.getElementById('summarizeInput').innerText = '';
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('audioStatus').textContent = 'Click "Start" to begin transcribing audio.';
            
            if (document.getElementById('chatContainer').style.display === 'flex') {
                document.getElementById('initialContainer').style.display = 'flex';
                document.getElementById('chatContainer').style.display = 'none';
            }
        }

        async function showChatHistory() {
            const response = await fetch('/chat-history');
            const chatHistory = await response.json();

            const chatHistoryList = document.getElementById('chatHistoryList');
            chatHistoryList.innerHTML = '';

            chatHistory.forEach(chat => {
                const chatItem = document.createElement('div');
                chatItem.className = 'chat-item';
                chatItem.innerHTML = `
                    <button onclick="loadChat('${chat.chat_id}')" aria-label="Load chat ${chat.chat_id}">
                        Chat ID: ${chat.chat_id}
                    </button>
                    <button class="delete-btn" onclick="deleteChat('${chat.chat_id}', this)" aria-label="Delete chat ${chat.chat_id}">
                        <i class="fas fa-trash"></i>
                    </button>
                `;
                chatHistoryList.appendChild(chatItem);
            });

            document.getElementById('chatHistoryModal').style.display = 'block';
            document.getElementById('chatHistoryModal').classList.add('show');
        }

        async function deleteChat(chatId, button) {
            if (!confirm('Are you sure you want to delete this chat?')) return;

            try {
                const response = await fetch('/delete-chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ chat_id: chatId })
                });
                const result = await response.json();

                if (result.message) {
                    alert('Chat deleted successfully!');
                    const chatItem = button.closest('.chat-item');
                    if (chatItem) chatItem.remove();
                } else {
                    alert('Error deleting chat: ' + (result.error || 'Unknown error'));
                }
            } catch (error) {
                console.error('Error deleting chat:', error);
                alert('Failed to delete chat.');
            }
        }

        function hideChatHistory() {
            const modal = document.getElementById('chatHistoryModal');
            modal.classList.remove('show');
            setTimeout(() => modal.style.display = 'none', 300);
        }

        async function loadChat(chatId) {
            const response = await fetch('/load-chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chat_id: chatId })
            });

            const chat = await response.json();
            currentChatId = chat.chat_id;

            if (chat.error) {
                alert(chat.error);
                return;
            }

            switchToChatUI();
            const chatOutput = document.getElementById('chatOutput');
            chatOutput.innerHTML = '';
            chat.messages.forEach(message => {
                const type = message.role === 'user' ? 'user' : 'system';
                const prefix = message.role === 'user' ? '<strong>You:</strong> ' : '<strong>Assistant:</strong> ';
                const content = message.content.startsWith('Audio transcription:') 
                    ? message.content 
                    : (message.content.startsWith('<h3>') ? message.content : `${prefix}${message.content}`);
                addMessage(content, type);
            });

            hideChatHistory();
        }

        function showSummarizeModal() {
            document.getElementById('summarizeModal').style.display = 'block';
            document.getElementById('summarizeModal').classList.add('show');
            document.getElementById('modalSummarizeInput').focus();
        }

        function hideSummarizeModal() {
            const modal = document.getElementById('summarizeModal');
            modal.classList.remove('show');
            setTimeout(() => modal.style.display = 'none', 300);
        }

        window.onclick = function(event) {
            const summarizeModal = document.getElementById('summarizeModal');
            const chatHistoryModal = document.getElementById('chatHistoryModal');
            if (event.target === summarizeModal) hideSummarizeModal();
            if (event.target === chatHistoryModal) hideChatHistory();
        }

        document.getElementById('chatOutput').addEventListener('click', function(e) {
            const exportBtn = e.target.closest('.export-btn');
            if (exportBtn) {
                const messageDiv = exportBtn.closest('.message');
                exportSingleMessage(messageDiv);
            }
        });

        async function exportSingleMessage(messageDiv) {
            try {
                const clone = messageDiv.cloneNode(true);
                clone.querySelector('.export-btn').remove();
                
                const container = document.createElement('div');
                container.style.backgroundColor = '#ffffff';
                container.style.color = '#000000';
                container.style.padding = '20px';
                container.style.maxWidth = '800px';
                container.style.position = 'absolute';
                container.style.left = '-9999px';
                
                const contentEl = clone.querySelector('.message-content');
                container.appendChild(contentEl.cloneNode(true));
                document.body.appendChild(container);

                const canvas = await html2canvas(container, {
                    scale: 2,
                    useCORS: true,
                    backgroundColor: '#ffffff'
                });

                const imgData = canvas.toDataURL('image/jpeg', 1.0);
                const pdf = new jspdf.jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

                pdf.setFontSize(18);
                pdf.text('AI Summary', 10, 10);
                pdf.setFontSize(10);
                pdf.text(`Generated on ${new Date().toLocaleDateString()}`, 10, 20);

                const imgWidth = 190;
                const imgHeight = (canvas.height / canvas.width) * imgWidth;
                pdf.addImage(imgData, 'JPEG', 10, 25, imgWidth, imgHeight);

                pdf.save('ai-summary.pdf');
                document.body.removeChild(container);
            } catch (error) {
                console.error('PDF generation error:', error);
                alert('Error generating PDF');
            }
        }