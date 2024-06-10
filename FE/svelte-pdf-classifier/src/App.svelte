<script>
    import { onMount } from 'svelte';
    let file;
    let result = null;
    let loading = false;

    async function classifyPDF() {
        if (!file) {
            alert("Please select a PDF file first.");
            return;
        }

        loading = true;
        result = null;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/classify/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to classify PDF');
            }

            const data = await response.json();
            result = data.predicted_category;
        } catch (error) {
            console.error(error);
            alert('An error occurred while classifying the PDF.');
        } finally {
            loading = false;
        }
    }
</script>

<style>
    .container {
        max-width: 600px;
        margin: 0 auto;
        padding: 2rem;
        text-align: center;
    }
    .file-input {
        margin-bottom: 1rem;
    }
    .result {
        margin-top: 1rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>

<div class="container">
    <h1>PDF Classifier</h1>
    <input class="file-input" type="file" accept="application/pdf" on:change="{e => file = e.target.files[0]}" />
    <button on:click="{classifyPDF}" disabled="{loading}">Classify PDF</button>
    {#if loading}
        <p>Loading...</p>
    {/if}
    {#if result !== null}
        <p class="result">Predicted Class: {result}</p>
    {/if}
</div>