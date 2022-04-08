% makeaxis.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to make phyplot like axis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      usage: makeaxis.m()
%         by: mehrdad jazayeri
%       date: Oct 2006
%
function makeaxis(varargin)

eval(evalargs(varargin));
if ~exist('majorTickRatio'), majorTickRatio = 0.01; end
if ~exist('minorTickRatio'), minorTickRatio = 0.01/2; end
if ~exist('offsetRatio'), offsetRatio = 0.04; end
if ~exist('x_label'), x_label=''; end
if ~exist('y_label'), y_label=''; end
if ~exist('xytitle'), xytitle=''; end

% turn off current axis
%axis tight;
axis off;
% get the current x and y limits
xlims = xlim;
ylims = ylim;

% get the current x and y tick positions
xticks = get(gca,'XTick');
yticks = get(gca,'YTick');
% get the current x and y tick labels
xticklabels = get(gca,'XTickLabel');
yticklabels = get(gca,'YTickLabel');
% get the current x and y labels
xaxis.label = x_label;
yaxis.label = y_label;
% get the current axis title
xaxis.xytitle = xytitle;

% set majotTickLen
xaxis.majorTickLen = majorTickRatio*(ylims(2)-ylims(1));
yaxis.majorTickLen = majorTickRatio*(xlims(2)-xlims(1));

% set minorTickLen
xaxis.minorTickLen = minorTickRatio*(ylims(2)-ylims(1));
yaxis.minorTickLen = minorTickRatio*(xlims(2)-xlims(1));

% set offset
xaxis.offset = offsetRatio*(ylims(2)-ylims(1));
yaxis.offset = offsetRatio*(xlims(2)-xlims(1));

axis([xlims ylims]+[-yaxis.offset-20*yaxis.majorTickLen 0 -xaxis.offset-20*xaxis.majorTickLen 0]);

% draw horizontal axis lines 
plot(xlims,[ylims(1)-xaxis.offset ylims(1)-xaxis.offset],'k');hold on

% draw major tick on horizontal axis with approporiate labels
for i = xticks
  thisticklabel = xticklabels(find(xticks==i),:);
  % draw major tick
  plot([i i],[ylims(1)-xaxis.offset ylims(1)-xaxis.majorTickLen-xaxis.offset],'k');
  % put label
  thandle = text(i,ylims(1)-xaxis.offset-1.5*xaxis.majorTickLen,thisticklabel);
%  get(thandle)
  % and format the text
  set(thandle,'HorizontalAlignment','center');
  set(thandle,'VerticalAlignment','top');
  set(thandle,'FontSize',10);
  set(thandle,'FontName','helvetica');
  set(thandle,'FontAngle','italic');
end

% draw vertical axis lines 
plot([xlims(1)-yaxis.offset xlims(1)-yaxis.offset],ylims,'k');hold on

% draw major tick on horizontal axis with approporiate labels
for i = yticks
  thisticklabel = yticklabels(find(yticks==i),:);
  % draw major tick
  plot([xlims(1)-yaxis.offset xlims(1)-yaxis.offset-yaxis.majorTickLen],[i i],'k');
  % draw text
  thandle = text(xlims(1)-yaxis.offset-2*yaxis.majorTickLen,i,thisticklabel);
  % and format the text
  set(thandle,'HorizontalAlignment','right');
  set(thandle,'VerticalAlignment','middle');
  set(thandle,'FontSize',10);
  set(thandle,'FontName','helvetica');
  set(thandle,'FontAngle','italic');
end

% add x axis label
thandle = text(mean(xlims(:)),ylims(1)-xaxis.offset-10*xaxis.majorTickLen,xaxis.label);
set(thandle,'HorizontalAlignment','center');
set(thandle,'VerticalAlignment','top');
set(thandle,'FontSize',8);
set(thandle,'FontName','helvetica');
set(thandle,'FontAngle','italic');
  
% add y axis label
thandle = text(xlims(1)-yaxis.offset-10*yaxis.majorTickLen,mean(ylims(:)),yaxis.label);
set(thandle,'HorizontalAlignment','center');
set(thandle,'VerticalAlignment','bottom');
set(thandle,'FontSize',10);
set(thandle,'FontName','helvetica');
set(thandle,'FontAngle','italic');
set(thandle,'Rotation',90);

% add title
thandle = text(mean(xlims(:)),ylims(2)+.02*(ylims(2)-ylims(1)),xaxis.xytitle);
set(thandle,'HorizontalAlignment','left');
set(thandle,'VerticalAlignment','bottom');
set(thandle,'FontSize',15);
set(thandle,'FontName','helvetica');
set(thandle,'FontAngle','italic');
%axis off;
