## GitHub Copilot Chat

- Extension Version: 0.28.5 (prod)
- VS Code: vscode/1.101.2
- OS: Mac

## Network

User Settings:

```json
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:

- DNS ipv4 Lookup: 140.82.113.6 (1 ms)
- DNS ipv6 Lookup: ::ffff:140.82.113.6 (2 ms)
- Proxy URL: None (0 ms)
- Electron fetch (configured): HTTP 200 (34 ms)
- Node.js https: HTTP 200 (123 ms)
- Node.js fetch: HTTP 200 (200 ms)
- Helix fetch: HTTP 200 (311 ms)

Connecting to https://api.githubcopilot.com/_ping:

- DNS ipv4 Lookup: 140.82.114.21 (148 ms)
- DNS ipv6 Lookup: ::ffff:140.82.113.22 (2 ms)
- Proxy URL: None (6 ms)
- Electron fetch (configured): HTTP 200 (132 ms)
- Node.js https: HTTP 200 (189 ms)
- Node.js fetch: HTTP 200 (102 ms)
- Helix fetch: HTTP 200 (206 ms)

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).
