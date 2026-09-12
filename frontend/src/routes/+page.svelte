<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import type { Candidate, JobDescription, BiasAuditResult, WeightConfig } from '$lib/types';
	import { 
		fetchCandidates, 
		fetchPresets, 
		fetchBiasAudit, 
		switchPreset, 
		uploadResumeFile, 
		uploadJDFile, 
		exportShortlistMarkdown 
	} from '$lib/api';

	import Sidebar from '$lib/components/Sidebar.svelte';
	import TopBar from '$lib/components/TopBar.svelte';
	import WeightStrip from '$lib/components/WeightStrip.svelte';
	import CandidateGrid from '$lib/components/CandidateGrid.svelte';
	import CandidateInspector from '$lib/components/CandidateInspector.svelte';
	import CompareModal from '$lib/components/CompareModal.svelte';
	import BiasDrawer from '$lib/components/BiasDrawer.svelte';
	import CommandPalette from '$lib/components/CommandPalette.svelte';
	import JobDescriptionModal from '$lib/components/JobDescriptionModal.svelte';

	// State
	let candidates = $state<Candidate[]>([]);
	let activeJD = $state<JobDescription | null>(null);
	let presets = $state<Record<string, JobDescription>>({});
	let biasAudit = $state<BiasAuditResult | null>(null);

	let selectedId = $state<string | null>(null);
	let shortlistedIds = $state<Set<string>>(new Set());
	let searchQuery = $state<string>('');

	// Modal visibility
	let isCompareOpen = $state<boolean>(false);
	let isBiasOpen = $state<boolean>(false);
	let isCommandPaletteOpen = $state<boolean>(false);
	let isJDModalOpen = $state<boolean>(false);
	let isWeightsOpen = $state<boolean>(true);
	let isSidebarCollapsed = $state<boolean>(false);
	let isInspectorCollapsed = $state<boolean>(false);
	let isDragging = $state<boolean>(false);
	let statusMessage = $state<string | null>(null);

	// Weight Configuration
	let config = $state<WeightConfig>({
		w_kw: 35,
		w_sem: 40,
		w_exp: 15,
		w_edu: 5,
		w_cert: 5,
		mode: 'hybrid',
		name_blind: false,
		strict_knockouts: false,
	});

	// Derived
	let selectedCandidate = $derived(
		candidates.find(c => c.candidate_id === selectedId) || (candidates[0] ?? null)
	);

	let filteredCandidates = $derived(
		candidates.filter(c => {
			if (!searchQuery.trim()) return true;
			const q = searchQuery.toLowerCase();
			return (
				c.name.toLowerCase().includes(q) ||
				c.matched_skills.some(s => s.toLowerCase().includes(q)) ||
				c.missing_skills.some(s => s.toLowerCase().includes(q))
			);
		})
	);

	let topFitCount = $derived(
		candidates.filter(c => c.final_score >= 0.80).length
	);

	let knockedOutCount = $derived(
		candidates.filter(c => c.is_knocked_out).length
	);

	onMount(() => {
		reloadAll();
		window.addEventListener('keydown', handleGlobalKeydown);
	});

	onDestroy(() => {
		if (typeof window !== 'undefined') {
			window.removeEventListener('keydown', handleGlobalKeydown);
		}
	});

	async function reloadAll() {
		try {
			const [candData, presetData, biasData] = await Promise.all([
				fetchCandidates(config),
				fetchPresets(),
				fetchBiasAudit()
			]);
			candidates = candData.candidates;
			activeJD = candData.jd;
			presets = presetData.presets;
			biasAudit = biasData;

			if (!selectedId && candidates.length > 0) {
				selectedId = candidates[0].candidate_id;
			}
		} catch (err: any) {
			console.error('Error initializing Crucible data:', err);
		}
	}

	async function handleWeightsChanged() {
		try {
			const res = await fetchCandidates(config);
			candidates = res.candidates;
			activeJD = res.jd;
		} catch (err: any) {
			console.error('Error re-scoring candidates:', err);
		}
	}

	async function handleSelectPreset(presetId: string) {
		try {
			const res = await switchPreset(presetId);
			activeJD = res.active_jd;
			candidates = res.candidates;
			selectedId = candidates[0]?.candidate_id || null;
			biasAudit = await fetchBiasAudit();
			showToast(`Switched to preset: ${activeJD.title}`);
		} catch (err: any) {
			showToast(`Failed to switch preset: ${err.message}`);
		}
	}

	async function handleUploadResume(file: File) {
		try {
			showToast(`Parsing and vectorizing ${file.name}...`);
			const res = await uploadResumeFile(file);
			candidates = res.candidates;
			selectedId = res.uploaded_id;
			showToast(`Ingested ${file.name} successfully`);
		} catch (err: any) {
			showToast(`Resume upload error: ${err.message}`);
		}
	}

	async function handleUploadJD(file: File) {
		try {
			showToast(`Analyzing custom JD ${file.name}...`);
			const res = await uploadJDFile(file);
			activeJD = res.active_jd;
			candidates = res.candidates;
			selectedId = candidates[0]?.candidate_id || null;
			biasAudit = await fetchBiasAudit();
			showToast(`Activated custom JD: ${file.name}`);
		} catch (err: any) {
			showToast(`JD upload error: ${err.message}`);
		}
	}

	async function handleJDSaved(newJD: JobDescription, newCandidates: Candidate[]) {
		activeJD = newJD;
		candidates = newCandidates;
		selectedId = candidates[0]?.candidate_id || null;
		biasAudit = await fetchBiasAudit();
		showToast(`Activated Job Spec: ${newJD.title}`);
	}

	async function handleExportShortlist() {
		try {
			const md = await exportShortlistMarkdown();
			const blob = new Blob([md], { type: 'text/markdown' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `crucible_shortlist_${activeJD?.id || 'export'}.md`;
			a.click();
			URL.revokeObjectURL(url);
			showToast('Exported shortlist to Markdown');
		} catch (err: any) {
			showToast(`Export error: ${err.message}`);
		}
	}

	function toggleShortlist(id: string) {
		const next = new Set(shortlistedIds);
		if (next.has(id)) next.delete(id);
		else next.add(id);
		shortlistedIds = next;
	}

	function showToast(msg: string) {
		statusMessage = msg;
		setTimeout(() => { if (statusMessage === msg) statusMessage = null; }, 3500);
	}

	// Desktop Keyboard Navigation Handler
	function handleGlobalKeydown(e: KeyboardEvent) {
		const isInput = ['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement)?.tagName);

		// Command Palette (Cmd+K / Ctrl+K)
		if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
			e.preventDefault();
			isCommandPaletteOpen = !isCommandPaletteOpen;
			return;
		}

		// Toggle Sidebar (Cmd+B / Ctrl+B)
		if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
			e.preventDefault();
			isSidebarCollapsed = !isSidebarCollapsed;
			return;
		}

		// Open Job Description Modal (Cmd+J / Ctrl+J)
		if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'j') {
			e.preventDefault();
			isJDModalOpen = !isJDModalOpen;
			return;
		}

		if (isCommandPaletteOpen || isCompareOpen || isBiasOpen || isJDModalOpen) {
			if (e.key === 'Escape') {
				isCommandPaletteOpen = false;
				isCompareOpen = false;
				isBiasOpen = false;
				isJDModalOpen = false;
			}
			return;
		}

		if (isInput) return;

		// [: Toggle Sidebar
		if (e.key === '[') {
			e.preventDefault();
			isSidebarCollapsed = !isSidebarCollapsed;
			return;
		}

		// ]: Toggle Candidate Inspector Aside
		if (e.key === ']') {
			e.preventDefault();
			isInspectorCollapsed = !isInspectorCollapsed;
			return;
		}

		// J / Down: Navigate down candidate list
		if (e.key === 'j' || e.key === 'ArrowDown') {
			e.preventDefault();
			navigateCandidates(1);
		}
		// K / Up: Navigate up candidate list
		else if (e.key === 'k' || e.key === 'ArrowUp') {
			e.preventDefault();
			navigateCandidates(-1);
		}
		// Space / Enter: Toggle shortlist
		else if (e.key === ' ' || e.key === 'Enter') {
			e.preventDefault();
			if (selectedId) toggleShortlist(selectedId);
		}
		// C: Head-to-Head Compare
		else if (e.key.toLowerCase() === 'c') {
			e.preventDefault();
			isCompareOpen = true;
		}
		// B: Toggle Name-Blind Mode
		else if (e.key.toLowerCase() === 'b') {
			e.preventDefault();
			config.name_blind = !config.name_blind;
			handleWeightsChanged();
		}
		// W: Toggle Weights Bar
		else if (e.key.toLowerCase() === 'w') {
			e.preventDefault();
			isWeightsOpen = !isWeightsOpen;
		}
		// /: Focus search
		else if (e.key === '/') {
			e.preventDefault();
			const el = document.querySelector('input[placeholder*="Filter"]') as HTMLInputElement;
			el?.focus();
		}
	}

	function navigateCandidates(delta: number) {
		if (filteredCandidates.length === 0) return;
		const currIdx = filteredCandidates.findIndex(c => c.candidate_id === selectedId);
		const nextIdx = Math.max(0, Math.min(filteredCandidates.length - 1, (currIdx === -1 ? 0 : currIdx) + delta));
		selectedId = filteredCandidates[nextIdx].candidate_id;
	}

	// Drag and Drop
	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		isDragging = true;
	}

	function handleDragLeave() {
		isDragging = false;
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		isDragging = false;
		if (e.dataTransfer?.files && e.dataTransfer.files[0]) {
			handleUploadResume(e.dataTransfer.files[0]);
		}
	}
</script>

<div class="h-screen w-screen overflow-hidden flex flex-col bg-white font-sans antialiased text-neutral-900">
	<!-- Drag & Drop PDF Overlay -->
	{#if isDragging}
		<div class="fixed inset-0 z-50 bg-neutral-900/80 text-white flex flex-col items-center justify-center p-8 border-4 border-dashed border-white">
			<div class="text-2xl font-bold font-mono">DROP PDF RESUME TO INGEST & RANK</div>
			<div class="text-sm font-mono text-neutral-300 mt-2">Zero-LLM instant CPU extraction & MiniLM vectorization</div>
		</div>
	{/if}

	<!-- 3-Pane Layout -->
	<div class="flex-1 flex overflow-hidden">
		<!-- Left Rail: Sidebar -->
		<Sidebar 
			activeJD={activeJD}
			presets={presets}
			biasAudit={biasAudit}
			candidateCount={candidates.length}
			topFitCount={topFitCount}
			knockedOutCount={knockedOutCount}
			isCollapsed={isSidebarCollapsed}
			onToggleCollapse={() => isSidebarCollapsed = !isSidebarCollapsed}
			onSelectPreset={handleSelectPreset}
			onOpenBiasDrawer={() => isBiasOpen = true}
			onOpenJDModal={() => isJDModalOpen = true}
			onUploadResume={handleUploadResume}
			onUploadJD={handleUploadJD}
		/>

		<!-- Center Pane: Data Grid & Controls -->
		<main class="flex-1 flex flex-col min-w-0 bg-white">
			<TopBar 
				bind:config={config}
				bind:searchQuery={searchQuery}
				activeCandidateName={selectedCandidate ? (config.name_blind ? `Candidate #${selectedCandidate.candidate_id.replace('cand_', '')}` : selectedCandidate.name) : undefined}
				isWeightsOpen={isWeightsOpen}
				onChange={handleWeightsChanged}
				onToggleWeights={() => isWeightsOpen = !isWeightsOpen}
				onOpenCompare={() => isCompareOpen = true}
				onOpenCommandPalette={() => isCommandPaletteOpen = true}
				onExportShortlist={handleExportShortlist}
			/>

			<WeightStrip 
				bind:config={config}
				isOpen={isWeightsOpen}
				onChange={handleWeightsChanged}
				onClose={() => isWeightsOpen = false}
			/>

			<CandidateGrid 
				candidates={filteredCandidates}
				selectedId={selectedId}
				shortlistedIds={shortlistedIds}
				isBlind={config.name_blind}
				isStrictKnockouts={config.strict_knockouts}
				onSelectCandidate={(c) => selectedId = c.candidate_id}
				onToggleShortlist={toggleShortlist}
				onCompareCandidate={(c) => { selectedId = c.candidate_id; isCompareOpen = true; }}
			/>
		</main>

		<!-- Right Pane: Telemetry Inspector -->
		<CandidateInspector 
			candidate={selectedCandidate}
			isShortlisted={selectedCandidate ? shortlistedIds.has(selectedCandidate.candidate_id) : false}
			isBlind={config.name_blind}
			isCollapsed={isInspectorCollapsed}
			onToggleCollapse={() => isInspectorCollapsed = !isInspectorCollapsed}
			onToggleShortlist={() => { if (selectedCandidate) toggleShortlist(selectedCandidate.candidate_id); }}
			onCompare={() => isCompareOpen = true}
		/>
	</div>

	<!-- Status Bar Footer -->
	<footer class="h-6 border-t border-neutral-200 bg-neutral-50 px-3 flex items-center justify-between font-mono text-[10px] text-neutral-500 select-none">
		<div class="flex items-center gap-2.5">
			<span class="flex items-center gap-1.5 font-semibold text-neutral-700">
				<span class="w-1.5 h-1.5 bg-emerald-600 rounded-full"></span>
				<span>ZERO-LLM ENGINE</span>
			</span>
			<span>•</span>
			<span>CPU LATENCY: ~34ms</span>
		</div>

		{#if statusMessage}
			<div class="text-neutral-900 font-semibold px-2 bg-neutral-200 truncate max-w-sm">
				{statusMessage}
			</div>
		{/if}

		<div class="flex items-center gap-3 text-neutral-500">
			<span><kbd class="px-1 border border-neutral-300 bg-white font-bold">J/K</kbd> Nav</span>
			<span><kbd class="px-1 border border-neutral-300 bg-white font-bold">Space</kbd> Shortlist</span>
			<span><kbd class="px-1 border border-neutral-300 bg-white font-bold">C</kbd> Compare</span>
			<span><kbd class="px-1 border border-neutral-300 bg-white font-bold">B</kbd> Blind</span>
			<span><kbd class="px-1 border border-neutral-300 bg-white font-bold">W</kbd> Weights</span>
			<span><kbd class="px-1 border border-neutral-300 bg-white font-bold">^J</kbd> Spec</span>
			<span><kbd class="px-1 border border-neutral-300 bg-white font-bold">⌘K</kbd> Palette</span>
		</div>
	</footer>

	<!-- Overlays & Modals -->
	<CompareModal 
		isOpen={isCompareOpen}
		candidates={candidates}
		selectedCandidate={selectedCandidate}
		onClose={() => isCompareOpen = false}
	/>

	<BiasDrawer 
		isOpen={isBiasOpen}
		audit={biasAudit}
		onClose={() => isBiasOpen = false}
	/>

	<CommandPalette 
		isOpen={isCommandPaletteOpen}
		candidates={candidates}
		presets={presets}
		onClose={() => isCommandPaletteOpen = false}
		onSelectCandidate={(c) => selectedId = c.candidate_id}
		onSelectPreset={handleSelectPreset}
		onOpenCompare={() => isCompareOpen = true}
		onOpenBias={() => isBiasOpen = true}
		onOpenJDModal={() => isJDModalOpen = true}
		onToggleBlind={() => { config.name_blind = !config.name_blind; handleWeightsChanged(); }}
		onToggleKnockouts={() => { config.strict_knockouts = !config.strict_knockouts; handleWeightsChanged(); }}
		onToggleSidebar={() => isSidebarCollapsed = !isSidebarCollapsed}
		onToggleInspector={() => isInspectorCollapsed = !isInspectorCollapsed}
		onToggleWeights={() => isWeightsOpen = !isWeightsOpen}
		onExport={handleExportShortlist}
	/>

	<JobDescriptionModal 
		isOpen={isJDModalOpen}
		activeJD={activeJD}
		presets={presets}
		onClose={() => isJDModalOpen = false}
		onJDSaved={handleJDSaved}
	/>
</div>
