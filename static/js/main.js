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
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': csrf_token
            },
            body: new URLSearchParams({
                'question': question,
                'video_id': videoId
            })
        });

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
        answerText.textContent = `エラー: ${error.message}`;
        answerText.classList.add('text-danger');
    }
}

// Add loading indicator for form submission
document.querySelector('form')?.addEventListener('submit', function(e) {
    const submitButton = this.querySelector('button[type="submit"]');
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 処理中...';
});
