import React, { useState, useEffect, useCallback } from 'react';
import { Upload, FileText, Database, Settings, Play, CheckCircle, AlertCircle, Clock, Download, Eye, Trash2, RefreshCw, Search, Plus, X, Monitor, Activity, Zap } from 'lucide-react';

const SakthiPlatform = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [processingSessions, setProcessingSessions] = useState([]);
  const [systemConnections, setSystemConnections] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentSession, setCurrentSession] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [systemMetrics, setSystemMetrics] = useState({});

  // Mock data initialization
  useEffect(() => {
    setSystemConnections([
      { id: 1, name: 'DeepSeek LLM', type: 'llm', status: 'connected', endpoint: 'http://llm-loadbalancer.local:80', lastSync: '2024-06-20 11:30' },
      { id: 2, name: 'Oracle HR DB', type: 'oracle', status: 'connected', lastSync: '2024-06-20 10:30' },
      { id: 3, name: 'BigQuery DW', type: 'bigquery', status: 'connected', lastSync: '2024-06-20 09:15' },
      { id: 4, name: 'SerpAPI', type: 'api', status: 'connected', lastSync: '2024-06-20 11:00' },
      { id: 5, name: 'ChromaDB', type: 'vector', status: 'connected', lastSync: '2024-06-20 11:25' }
    ]);

    setProcessingSessions([
      {
        id: 'session_abc123',
        name: 'Oracle to BigQuery Migration',
        status: 'completed',
        progress: 100,
        startTime: '2024-06-20 09:00',
        endTime: '2024-06-20 09:45',
        qualityScore: 0.92,
        confidence: 0.89,
        files: ['hr_schema.sql', 'employee_data.csv'],
        type: 'schema_migration'
      },
      {
        id: 'session_def456',
        name: 'PDF Data Extraction',
        status: 'processing',
        progress: 65,
        startTime: '2024-06-20 10:30',
        endTime: null,
        qualityScore: null,
        confidence: null,
        files: ['financial_report.pdf'],
        type: 'document_processing'
      },
      {
        id: 'session_ghi789',
        name: 'Web Data Scraping',
        status: 'completed',
        progress: 100,
        startTime: '2024-06-20 08:15',
        endTime: '2024-06-20 08:45',
        qualityScore: 0.85,
        confidence: 0.78,
        files: ['competitor_data.json'],
        type: 'web_scraping'
      }
    ]);

    setSystemMetrics({
      totalSessions: 24,
      activeSessions: 3,
      completedToday: 8,
      avgProcessingTime: '2.3 min',
      llmCalls: 156,
      successRate: 94.2
    });
  }, []);

  const handleFileUpload = useCallback((event) => {
    const files = Array.from(event.target.files);
    const newFiles = files.map(file => ({
      id: Date.now() + Math.random(),
      name: file.name,
      size: file.size,
      type: file.type,
      uploadTime: new Date().toISOString(),
      status: 'uploaded',
      file: file
    }));
    
    setUploadedFiles(prev => [...prev, ...newFiles]);
  }, []);

  const startProcessing = useCallback((intent) => {
    if (uploadedFiles.length === 0 && !intent.includes('web') && !intent.includes('url')) {
      alert('Please upload files first or specify a web URL');
      return;
    }

    setIsProcessing(true);
    const sessionId = `session_${Date.now()}`;
    
    const newSession = {
      id: sessionId,
      name: intent || 'Custom Processing',
      status: 'processing',
      progress: 0,
      startTime: new Date().toISOString(),
      endTime: null,
      qualityScore: null,
      confidence: null,
      files: uploadedFiles.map(f => f.name),
      type: 'custom'
    };

    setProcessingSessions(prev => [newSession, ...prev]);
    setCurrentSession(newSession);

    // Simulate processing with realistic progress
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 20;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        setIsProcessing(false);
        
        // Update session as completed
        setProcessingSessions(prev => 
          prev.map(session => 
            session.id === sessionId 
              ? { 
                  ...session, 
                  status: 'completed', 
                  progress: 100,
                  endTime: new Date().toISOString(),
                  qualityScore: 0.75 + Math.random() * 0.25,
                  confidence: 0.70 + Math.random() * 0.30
                }
              : session
          )
        );
        setCurrentSession(null);
        
        // Update metrics
        setSystemMetrics(prev => ({
          ...prev,
          totalSessions: prev.totalSessions + 1,
          completedToday: prev.completedToday + 1,
          llmCalls: prev.llmCalls + Math.floor(Math.random() * 10) + 5
        }));
      } else {
        setProcessingSessions(prev => 
          prev.map(session => 
            session.id === sessionId 
              ? { ...session, progress: Math.round(progress) }
              : session
          )
        );
      }
    }, 800);
  }, [uploadedFiles]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'processing': return <Clock className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'failed': return <AlertCircle className="w-4 h-4 text-red-500" />;
      default: return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getConnectionStatusColor = (status) => {
    switch (status) {
      case 'connected': return 'bg-green-100 text-green-800 border-green-200';
      case 'disconnected': return 'bg-red-100 text-red-800 border-red-200';
      case 'connecting': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'llm': return <Zap className="w-4 h-4" />;
      case 'oracle': case 'bigquery': case 'vector': return <Database className="w-4 h-4" />;
      case 'api': return <Activity className="w-4 h-4" />;
      default: return <Monitor className="w-4 h-4" />;
    }
  };

  const filteredSessions = processingSessions.filter(session =>
    session.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    session.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-lg border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <Database className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Sakthi Platform</h1>
                <p className="text-purple-300 text-sm">AI-Powered Data Processing with DeepSeek LLM</p>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="text-center">
                  <div className="text-xl font-bold text-white">{systemMetrics.totalSessions}</div>
                  <div className="text-purple-300">Total Sessions</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-green-400">{systemMetrics.successRate}%</div>
                  <div className="text-purple-300">Success Rate</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-blue-400">{systemMetrics.llmCalls}</div>
                  <div className="text-purple-300">LLM Calls</div>
                </div>
              </div>
              <button className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors">
                <Settings className="w-4 h-4 inline mr-2" />
                Settings
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex space-x-1 bg-black/20 backdrop-blur-lg rounded-xl p-1 mb-8">
          {[
            { id: 'upload', label: 'Upload & Process', icon: Upload },
            { id: 'sessions', label: 'Processing Sessions', icon: Clock },
            { id: 'connections', label: 'System Connections', icon: Database },
            { id: 'metrics', label: 'Analytics', icon: Activity }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg transition-all ${
                activeTab === tab.id 
                  ? 'bg-purple-600 text-white shadow-lg' 
                  : 'text-purple-300 hover:text-white hover:bg-white/10'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Main Content */}
        <div className="space-y-6">
          {/* Upload & Process Tab */}
          {activeTab === 'upload' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* File Upload Section */}
              <div className="bg-black/20 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20">
                <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <Upload className="w-5 h-5 mr-2" />
                  Upload Documents
                </h2>
                
                <div className="border-2 border-dashed border-purple-500/30 rounded-lg p-8 text-center hover:border-purple-500/50 transition-colors">
                  <input
                    type="file"
                    multiple
                    onChange={handleFileUpload}
                    className="hidden"
                    id="file-upload"
                    accept=".pdf,.docx,.csv,.xlsx,.json,.txt,.sql"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <div className="w-16 h-16 mx-auto mb-4 bg-purple-600/20 rounded-full flex items-center justify-center">
                      <Upload className="w-8 h-8 text-purple-400" />
                    </div>
                    <p className="text-white font-medium mb-2">Drop files here or click to browse</p>
                    <p className="text-purple-300 text-sm">Supports PDF, DOCX, CSV, XLSX, JSON, TXT, SQL</p>
                  </label>
                </div>

                {uploadedFiles.length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-white font-medium mb-3">Uploaded Files ({uploadedFiles.length})</h3>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {uploadedFiles.map(file => (
                        <div key={file.id} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-purple-500/10">
                          <div className="flex items-center space-x-3">
                            <FileText className="w-4 h-4 text-purple-400" />
                            <div>
                              <p className="text-white text-sm font-medium">{file.name}</p>
                              <p className="text-purple-300 text-xs">{formatFileSize(file.size)}</p>
                            </div>
                          </div>
                          <button
                            onClick={() => setUploadedFiles(prev => prev.filter(f => f.id !== file.id))}
                            className="text-red-400 hover:text-red-300 transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Processing Controls */}
              <div className="bg-black/20 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20">
                <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <Play className="w-5 h-5 mr-2" />
                  Sakthi Processing
                </h2>

                <div className="space-y-4">
                  {/* Quick Actions */}
                  <div className="grid grid-cols-1 gap-3">
                    {[
                      'Extract data from uploaded documents',
                      'Convert Oracle schema to BigQuery',
                      'Transform CSV to structured JSON',
                      'Scrape data from competitor websites',
                      'Generate API endpoints from data',
                      'Create schema migration scripts'
                    ].map((action, index) => (
                      <button
                        key={index}
                        onClick={() => startProcessing(action)}
                        disabled={isProcessing}
                        className="text-left p-3 bg-white/5 hover:bg-white/10 rounded-lg border border-purple-500/20 hover:border-purple-500/40 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <p className="text-white text-sm font-medium">{action}</p>
                      </button>
                    ))}
                  </div>

                  {/* Custom Intent Input */}
                  <div className="space-y-3">
                    <label className="text-white text-sm font-medium">Custom Intent (Natural Language)</label>
                    <textarea
                      placeholder="Describe what you want to do with your data... 
Examples:
- Extract quarterly revenue from financial reports
- Map customer table from MySQL to PostgreSQL  
- Get pricing data from competitor.com daily
- Convert PDF invoices to structured JSON"
                      className="w-full p-3 bg-white/5 border border-purple-500/20 rounded-lg text-white placeholder-purple-300 focus:outline-none focus:border-purple-500/50 resize-none"
                      rows={4}
                      id="custom-intent"
                    />
                    <button
                      onClick={() => {
                        const intent = document.getElementById('custom-intent').value;
                        if (intent.trim()) {
                          startProcessing(intent);
                          document.getElementById('custom-intent').value = '';
                        }
                      }}
                      disabled={isProcessing}
                      className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white py-3 rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isProcessing ? (
                        <>
                          <RefreshCw className="w-4 h-4 inline mr-2 animate-spin" />
                          Processing with DeepSeek LLM...
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4 inline mr-2" />
                          Start Processing
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Current Processing Status */}
                {currentSession && (
                  <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-white font-medium">Processing: {currentSession.name}</p>
                      <span className="text-blue-400 text-sm">{currentSession.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${currentSession.progress}%` }}
                      />
                    </div>
                    <p className="text-blue-300 text-xs mt-2">
                      Files: {currentSession.files.length > 0 ? currentSession.files.join(', ') : 'Web processing'}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Processing Sessions Tab */}
          {activeTab === 'sessions' && (
            <div className="bg-black/20 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white flex items-center">
                  <Clock className="w-5 h-5 mr-2" />
                  Processing Sessions
                </h2>
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-purple-400" />
                    <input
                      type="text"
                      placeholder="Search sessions..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10 pr-4 py-2 bg-white/5 border border-purple-500/20 rounded-lg text-white placeholder-purple-300 focus:outline-none focus:border-purple-500/50"
                    />
                  </div>
                  <button className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors">
                    <RefreshCw className="w-4 h-4 inline mr-2" />
                    Refresh
                  </button>
                </div>
              </div>

              <div className="space-y-4">
                {filteredSessions.map(session => (
                  <div key={session.id} className="p-4 bg-white/5 rounded-lg border border-purple-500/20 hover:border-purple-500/40 transition-all">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(session.status)}
                        <div>
                          <h3 className="text-white font-medium">{session.name}</h3>
                          <p className="text-purple-300 text-sm">Session ID: {session.id}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        {session.confidence && (
                          <span className="bg-blue-500/20 text-blue-400 px-2 py-1 rounded text-xs">
                            Confidence: {(session.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                        {session.qualityScore && (
                          <span className="bg-green-500/20 text-green-400 px-2 py-1 rounded text-xs">
                            Quality: {(session.qualityScore * 100).toFixed(0)}%
                          </span>
                        )}
                        <span className={`px-2 py-1 rounded text-xs ${
                          session.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                          session.status === 'processing' ? 'bg-blue-500/20 text-blue-400' :
                          'bg-red-500/20 text-red-400'
                        }`}>
                          {session.status.charAt(0).toUpperCase() + session.status.slice(1)}
                        </span>
                      </div>
                    </div>

                    {session.status === 'processing' && (
                      <div className="mb-3">
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-purple-300">Progress</span>
                          <span className="text-white">{session.progress}%</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <div 
                            className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all"
                            style={{ width: `${session.progress}%` }}
                          />
                        </div>
                      </div>
                    )}

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-purple-300">Started</p>
                        <p className="text-white">{new Date(session.startTime).toLocaleString()}</p>
                      </div>
                      {session.endTime && (
                        <div>
                          <p className="text-purple-300">Completed</p>
                          <p className="text-white">{new Date(session.endTime).toLocaleString()}</p>
                        </div>
                      )}
                      <div>
                        <p className="text-purple-300">Files/Type</p>
                        <p className="text-white">{session.files.length > 0 ? `${session.files.length} files` : session.type}</p>
                      </div>
                      <div className="flex space-x-2">
                        <button className="bg-purple-600 hover:bg-purple-700 text-white px-3 py-1 rounded text-xs transition-colors">
                          <Eye className="w-3 h-3 inline mr-1" />
                          View
                        </button>
                        {session.status === 'completed' && (
                          <button className="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-xs transition-colors">
                            <Download className="w-3 h-3 inline mr-1" />
                            Export
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* System Connections Tab */}
          {activeTab === 'connections' && (
            <div className="bg-black/20 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white flex items-center">
                  <Database className="w-5 h-5 mr-2" />
                  System Connections
                </h2>
                <button className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors">
                  <Plus className="w-4 h-4 inline mr-2" />
                  Add Connection
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {systemConnections.map(connection => (
                  <div key={connection.id} className="p-4 bg-white/5 rounded-lg border border-purple-500/20 hover:border-purple-500/40 transition-all">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        {getTypeIcon(connection.type)}
                        <h3 className="text-white font-medium">{connection.name}</h3>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs border ${getConnectionStatusColor(connection.status)}`}>
                        {connection.status}
                      </span>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-purple-300">Type:</span>
                        <span className="text-white">{connection.type.toUpperCase()}</span>
                      </div>
                      {connection.endpoint && (
                        <div className="flex justify-between">
                          <span className="text-purple-300">Endpoint:</span>
                          <span className="text-white text-xs truncate max-w-32">{connection.endpoint}</span>
                        </div>
                      )}
                      <div className="flex justify-between">
                        <span className="text-purple-300">Last Sync:</span>
                        <span className="text-white text-xs">{connection.lastSync}</span>
                      </div>
                    </div>
                    <div className="flex space-x-2 mt-4">
                      <button className="flex-1 bg-purple-600 hover:bg-purple-700 text-white py-1 px-3 rounded text-xs transition-colors">
                        Test
                      </button>
                      <button className="flex-1 bg-gray-600 hover:bg-gray-700 text-white py-1 px-3 rounded text-xs transition-colors">
                        Configure
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Analytics Tab */}
          {activeTab === 'metrics' && (
            <div className="space-y-6">
              {/* Metrics Overview */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { label: 'Total Sessions', value: systemMetrics.totalSessions, icon: Clock, color: 'blue' },
                  { label: 'Active Sessions', value: systemMetrics.activeSessions, icon: Activity, color: 'green' },
                  { label: 'Completed Today', value: systemMetrics.completedToday, icon: CheckCircle, color: 'purple' },
                  { label: 'Success Rate', value: `${systemMetrics.successRate}%`, icon: Zap, color: 'pink' }
                ].map((metric, index) => (
                  <div key={index} className="bg-black/20 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-purple-300 text-sm">{metric.label}</p>
                        <p className="text-2xl font-bold text-white">{metric.value}</p>
                      </div>
                      <metric.icon className={`w-8 h-8 text-${metric.color}-400`} />
                    </div>
                  </div>
                ))}
              </div>

              {/* Performance Chart Area */}
              <div className="bg-black/20 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20">
                <h3 className="text-lg font-medium text-white mb-4">Processing Performance</h3>
                <div className="h-64 flex items-center justify-center border border-purple-500/20 rounded-lg">
                  <div className="text-center">
                    <Activity className="w-16 h-16 mx-auto text-purple-400 mb-4" />
                    <p className="text-purple-300">Performance charts would be rendered here</p>
                    <p className="text-sm text-purple-400 mt-2">Integration with Chart.js or D3 for real metrics</p>
                  </div>
                </div>
              </div>

              {/* Recent Activity */}
              <div className="bg-black/20 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20">
                <h3 className="text-lg font-medium text-white mb-4">Recent Activity</h3>
                <div className="space-y-3">
                  {[
                    { time: '2 min ago', action: 'Schema migration completed', user: 'System', type: 'success' },
                    { time: '5 min ago', action: 'New document uploaded: financial_report.pdf', user: 'User', type: 'info' },
                    { time: '12 min ago', action: 'LLM enhancement applied to session_abc123', user: 'DeepSeek', type: 'enhancement' },
                    { time: '18 min ago', action: 'Web scraping job started', user: 'SerpAPI', type: 'processing' },
                    { time: '25 min ago', action: 'ChromaDB vector storage updated', user: 'System', type: 'storage' }
                  ].map((activity, index) => (
                    <div key={index} className="flex items-center space-x-3 p-3 bg-white/5 rounded-lg">
                      <div className={`w-2 h-2 rounded-full ${
                        activity.type === 'success' ? 'bg-green-400' :
                        activity.type === 'info' ? 'bg-blue-400' :
                        activity.type === 'enhancement' ? 'bg-purple-400' :
                        activity.type === 'processing' ? 'bg-yellow-400' :
                        'bg-gray-400'
                      }`} />
                      <div className="flex-1">
                        <p className="text-white text-sm">{activity.action}</p>
                        <p className="text-purple-300 text-xs">{activity.user} • {activity.time}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SakthiPlatform;