# Google Cloud Vision API Setup Guide

## Why Use Google Cloud Vision?
- **Much better accuracy** for phone photos (handles blur, shadows, glare)
- **Handles complex layouts** better than local OCR
- **More reliable** for real-world images

## Setup Steps

### 1. Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Note your project ID

### 2. Enable Vision API
1. Go to **APIs & Services** > **Library**
2. Search for "Cloud Vision API"
3. Click **Enable**

### 3. Create Service Account
1. Go to **IAM & Admin** > **Service Accounts**
2. Click **Create Service Account**
3. Name it (e.g., "recipe-ocr")
4. Click **Create and Continue**
5. **Grant role** (you can skip this step - see note below):
   - In the role dropdown, search for "Vision" 
   - If you see **Cloud Vision API User** or similar, select it
   - **OR** you can skip role assignment here (it's often not needed if API is enabled)
   - **OR** use a basic role like **Viewer** or **Editor** if you want broader access
6. Click **Done** (or **Skip** if no role needed)

**Important Note**: 
- If the Vision API is enabled, the service account can usually use it without a specific role
- The key is having the **API enabled** (step 2) and a valid **JSON key** (step 4)
- You can always add roles later in **IAM & Admin** > **IAM** if needed

### 4. Create and Download Key
1. Click on the service account you just created
2. Go to **Keys** tab
3. Click **Add Key** > **Create new key**
4. Choose **JSON** format
5. Download the JSON file
6. **IMPORTANT**: Keep this file secure! Don't commit it to git.

### 5. Set Up Credentials
You have two options:

#### Option A: Environment Variable (Recommended)
1. Place the JSON file in your project directory (or secure location)
2. Add to your `.env` file:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
   ```
   Or for relative path:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json
   ```

#### Option B: Set Environment Variable Directly
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### 6. Install Package
```bash
pip install google-cloud-vision
```

### 7. Test It
The app will automatically use Google Vision if credentials are configured.
Check logs for: "Google Cloud Vision API initialized successfully"

## Pricing
- **First 1,000 units/month**: FREE
- **1,001 - 5,000,000 units**: $1.50 per 1,000 units
- **5,000,001+ units**: $0.60 per 1,000 units

1 image = 1 unit, so first 1,000 images per month are free!

## Security Notes
- **Never commit** the JSON key file to git
- Add `service-account-key.json` to `.gitignore`
- Use environment variables, not hardcoded paths
- Rotate keys periodically

## Troubleshooting

### "Credentials not found"
- Check that `GOOGLE_APPLICATION_CREDENTIALS` is set correctly
- Verify the JSON file path is correct
- Make sure the file exists and is readable

### "Permission denied"
- Check that the service account has "Cloud Vision API User" role
- Verify Vision API is enabled in your project

### "Quota exceeded"
- Check your usage in Google Cloud Console
- Upgrade billing if needed

## Fallback Behavior
If Google Vision is not configured or fails, the app will automatically fall back to:
1. PaddleOCR
2. EasyOCR  
3. Pytesseract

So your app will still work even without Google Vision!

