# 🚀 Streamlit Deployment Guide

## Quick Fixes for "Nothing Showing" Issue

### 1. **Check Your Entry Point**
Make sure your Streamlit deployment is pointing to the correct file:
- **Option A**: Deploy `streamlit_app.py` (recommended)
- **Option B**: Deploy `src/app/main.py` directly

### 2. **Verify API Keys in Streamlit Secrets**
Your app requires these secrets in Streamlit Cloud:

```toml
PINECONE_API_KEY = "your-pinecone-api-key"
GOOGLE_API_KEY = "your-google-api-key"
PINECONE_INDEX_NAME = "medical-rag-index"
PINECONE_NAMESPACE = "thera-rag"
```

### 3. **Check Deployment Logs**
1. Go to your Streamlit Cloud dashboard
2. Click on your app
3. Check the "Logs" tab for error messages

### 4. **Common Issues & Solutions**

#### ❌ **"Module not found" errors**
- Make sure `requirements.txt` is in the root directory
- Check that all dependencies are listed

#### ❌ **"API key not found" errors**
- Verify secrets are set in Streamlit Cloud
- Check secret names match exactly

#### ❌ **"Pinecone connection failed"**
- Verify Pinecone API key is correct
- Check if index name exists in your Pinecone account

#### ❌ **"Google API error"**
- Verify Google API key has proper permissions
- Enable necessary Google APIs (Generative AI, etc.)

### 5. **Test Locally First**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PINECONE_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Run locally
streamlit run streamlit_app.py
```

### 6. **Deployment Checklist**
- [ ] `requirements.txt` in root directory
- [ ] `.streamlit/config.toml` created
- [ ] API keys set in Streamlit secrets
- [ ] Pinecone index exists and is accessible
- [ ] Google API key has proper permissions
- [ ] All files committed to repository

### 7. **Still Not Working?**
1. Check the browser console for JavaScript errors
2. Try clearing browser cache
3. Check if the app is actually running (not just loading)
4. Verify your repository is properly connected to Streamlit Cloud

## Need Help?
If you're still having issues, check the deployment logs and look for specific error messages.
