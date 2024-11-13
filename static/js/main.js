async function checkProgress(videoId) {
    const progressBar = document.querySelector('.progress-bar');
    const statusText = document.querySelector('.status-text');
    let startTime = Date.now();
    
    while (true) {
        try {
            // Check for timeout (30 seconds)
            if (Date.now() - startTime > 30000) {
                statusText.textContent = "処理がタイムアウトしました。もう一度お試しください。";
                progressBar.classList.remove('progress-bar-animated');
                return;
            }

            const response = await fetch(`/progress/${videoId}`);
            if (!response.ok) {
                throw new Error('進行状況の取得に失敗しました');
            }

            const data = await response.json();
            
            if (data.status === 'processing') {
                const progress = data.progress;
                progressBar.style.width = `${progress.percentage}%`;
                progressBar.setAttribute('aria-valuenow', progress.percentage);
                progressBar.textContent = `${progress.percentage}%`;
                statusText.textContent = progress.status;
            } else if (data.status === 'completed') {
                window.location.href = data.redirect;
                return;
            } else if (data.status === 'error') {
                statusText.textContent = `エラー: ${data.error}`;
                progressBar.classList.remove('progress-bar-animated');
                return;
            }

            // Wait before next check
            await new Promise(resolve => setTimeout(resolve, 1000));
        } catch (error) {
            statusText.textContent = `エラー: ${error.message}`;
            progressBar.classList.remove('progress-bar-animated');
            return;
        }
    }
}

async function askQuestion() {
    const question = document.getElementById('question').value;
    const videoId = document.getElementById('video-id').value;
    const answerSection = document.getElementById('answer-section');
    const answerText = document.getElementById('answer-text');
    const csrf_token = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

    if (!question.trim()) {
        alert('質問を入力してください');
        return;
    }

    // Show loading state
    answerSection.style.display = 'block';
    answerText.innerHTML = '<div class="loading">回答を生成中</div>';

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': csrf_token
            },
            body: new URLSearchParams({
                'question': question,
                'video_id': videoId
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            if (response.status === 405) {
                throw new Error('無効なリクエストメソッドです');
            }
            const errorData = await response.json();
            throw new Error(errorData.error || '回答の生成に失敗しました');
        }

        const data = await response.json();
        answerText.textContent = data.answer;
    } catch (error) {
        if (error.name === 'AbortError') {
            answerText.textContent = '処理がタイムアウトしました。もう一度お試しください。';
        } else {
            answerText.textContent = `エラー: ${error.message}`;
        }
        answerText.classList.add('text-danger');
    }
}

// Add loading indicator for form submission
document.querySelector('form')?.addEventListener('submit', function(e) {
    const submitButton = this.querySelector('button[type="submit"]');
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 処理中...';
});

// Start progress checking if we're on the processing page
if (document.querySelector('.progress-section')) {
    const videoId = document.getElementById('video-id').value;
    checkProgress(videoId);
}
