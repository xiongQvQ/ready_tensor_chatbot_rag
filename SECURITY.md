# Security Policy

## Overview

This document outlines security practices, policies, and guidelines for the RAG-Powered Research Knowledge Assistant.

---

## 🔐 Security Measures

### 1. API Key Management

#### Best Practices

**✅ DO:**

- Store API keys in `.env` file (never in code)
- Use environment variables for sensitive data
- Keep `.env` in `.gitignore`
- Rotate keys periodically (every 90 days)
- Use separate keys for dev/prod environments

**❌ DON'T:**

- Commit API keys to version control
- Share keys in chat/email
- Hardcode keys in source code
- Use production keys in development
- Store keys in plaintext files

#### Key Storage

**Correct:**

```env
# .env file (gitignored)
GROQ_API_KEY=gsk_your_key_here
```

**Wrong:**

```python
# NEVER do this in code!
api_key = "gsk_your_key_here"
```

#### Key Rotation

To rotate your API key:

1. Generate new key at <https://console.groq.com/>
2. Update `.env` file
3. Test system with new key
4. Revoke old key
5. Update documentation

---

### 2. Data Privacy

#### Document Security

**Local Storage:**

- All documents stored locally in `documents/` folder
- Vector database stored locally in `research_db/`
- No documents sent to external services (except embeddings)

**Data Flow:**

1. Documents → Local processing → Local embeddings
2. Queries → Local embedding → API (query only, not documents)
3. API response → Local display

**What's sent to external services:**

- ✅ User queries (to Groq API)
- ✅ Text chunks for embedding (to HuggingFace, cached locally)
- ❌ Full documents (never sent)
- ❌ Vector database (stays local)

#### Sensitive Documents

For sensitive/confidential documents:

1. **Verify data handling**: Review Groq & HuggingFace privacy policies
2. **Use private embeddings**: Host sentence-transformers locally
3. **Consider self-hosted LLM**: Use local LLaMA instead of Groq
4. **Network isolation**: Run on air-gapped systems if required

---

### 3. Access Control

#### File Permissions

Recommended permissions:

```bash
chmod 600 .env                    # Only owner can read/write
chmod 644 documents/*.txt         # Owner write, others read
chmod 700 research_db/            # Only owner access
```

#### Multi-user Setup

For shared systems:

1. Use separate virtual environments per user
2. Individual `.env` files (not shared)
3. Separate `research_db/` directories
4. Document folder access controls

---

### 4. Input Validation

#### Current Measures

**Query Validation:**

- Length checks (min 3, max 500 characters)
- Character encoding validation (UTF-8)
- Basic sanitization of special characters

**Document Validation:**

- File type restrictions (.txt only)
- Encoding checks (UTF-8)
- Size limits (configurable)

#### Planned Enhancements (Phase 3)

- [ ] Advanced input sanitization
- [ ] SQL injection pattern detection
- [ ] Command injection prevention
- [ ] Rate limiting
- [ ] Query logging with anonymization

---

### 5. Dependency Security

#### Keeping Dependencies Updated

**Check for vulnerabilities:**

```bash
pip install safety
safety check -r requirements.txt
```

**Update packages:**

```bash
pip list --outdated
pip install --upgrade <package-name>
```

**Automated scanning:**

- GitHub Dependabot enabled
- Security alerts monitored
- Regular dependency updates

#### Known Dependencies

All dependencies pinned in `requirements.txt`:

- ChromaDB: 0.4.22
- LangChain: 0.2.0
- PyTorch: 2.0+
- Sentence-transformers: 2.2.2+

**Security updates:**
Monitor for security patches in:

- LangChain (high priority)
- ChromaDB (high priority)
- PyTorch (medium priority)

---

## 🚨 Reporting Vulnerabilities

### How to Report

If you discover a security vulnerability:

1. **DO NOT** create a public GitHub issue
2. Contact the project maintainers privately
3. Include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **24 hours**: Initial response acknowledgment
- **7 days**: Preliminary assessment
- **30 days**: Fix or mitigation (for confirmed issues)

### Disclosure Policy

- Coordinated disclosure preferred
- Public disclosure after fix released
- Credit given to reporter (unless anonymous)

---

## 🛡️ Security Best Practices

### For Production Deployment

1. **Environment Isolation**
   - Use separate dev/staging/prod environments
   - Different API keys per environment
   - Separate databases

2. **Logging & Monitoring**
   - Log all API calls (without sensitive data)
   - Monitor for unusual query patterns
   - Set up alerts for errors

3. **Rate Limiting**
   - Implement per-user query limits
   - Prevent API quota exhaustion
   - Protect against abuse

4. **Regular Audits**
   - Review access logs monthly
   - Check for outdated dependencies
   - Test backup/restore procedures

5. **Backup Strategy**
   - Regular backups of `research_db/`
   - Document storage backups
   - Configuration backups

---

## 📋 Security Checklist

Before deploying to production:

- [ ] API keys stored in `.env` (not in code)
- [ ] `.env` added to `.gitignore`
- [ ] File permissions set correctly
- [ ] Dependencies updated and scanned
- [ ] Input validation enabled
- [ ] Logging configured
- [ ] Backup strategy implemented
- [ ] Access controls defined
- [ ] Security policy documented
- [ ] Team trained on security practices

---

## 🔍 Audit Log

| Date | Change | Reason | Reviewer |
|------|--------|--------|----------|
| 2025-10-11 | Initial security policy | Project setup | Claude Code |
| - | - | - | - |

---

## 📚 Additional Resources

### Security Guidelines

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [LangChain Security](https://python.langchain.com/docs/security)

### API Provider Policies

- [Groq Privacy Policy](https://groq.com/privacy-policy/)
- [HuggingFace Privacy](https://huggingface.co/privacy)

### Compliance Resources

- GDPR compliance (for EU users)
- HIPAA guidance (for healthcare data)
- SOC 2 requirements (for enterprise)

---

## 📜 License

Security policy licensed under same terms as project (MIT License).

---

**Last Updated**: 2025-10-11
**Version**: 1.0
**Maintainer**: Project Team
